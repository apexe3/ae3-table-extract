# AE3-TL GPU service: deployment guide

The fast way to run AE3-TL in production: a Docker image that reproduces the benchmarked setup
exactly and serves tables over HTTP on port 8000.

| measured | result |
|---|---|
| one page with its words, warm (A100) | **190–210 ms** |
| 3 model copies on half of an A100, shared with a vLLM model, words attached | **37.7 pages/s, ~136,000 pages/hour**, 300 of 300 OK |
| same, structure only (no words) | 46.5 pages/s, ~167,000 pages/hour |
| 8 model copies on an RTX 5090, PubTables-v2 test pages | 70.0 pages/s, ~252,000 pages/hour |

Images on Docker Hub: **`apexe3/ae3-table-extract:gpu`** (NVIDIA CUDA, x86_64) and
**`apexe3/ae3-table-extract:cpu`** (x86_64 and ARM, includes Tesseract for scanned pages).
Versioned tags: `:0.24-gpu`, `:0.24-cpu`.

---

## What's in this folder

```
ae3-gpu-setup/
├── README.md              ← you are here
├── Dockerfile             ← GPU image: CUDA 12.4.1, Python 3.11, onnxruntime-gpu 1.23.2, no Tesseract
├── Dockerfile.cpu         ← CPU image: same code and pins, CPU onnxruntime, Tesseract included
├── docker/entrypoint.sh   ← start-up: key check, weights once, GPU check, AE3_COPIES workers
├── build_and_push.sh      ← builds both images and pushes them to Docker Hub
├── requirements.txt       ← the exact package versions the benchmark ran with
├── setup_gpu.sh           ← install without Docker (same packages, same checks)
├── serve.sh               ← start the no-Docker install
├── scripts/make_words.py  ← PDFs or scans → page images + words files, ready to send
├── scripts/bench.py       ← throughput and latency of a running service
└── ae3-tl-build/          ← the service source: API, model runtime, post-processing, tests
```

The HTTP API itself (endpoints, response shape, keys, limits) is documented in
[`ae3-tl-build/README.md`](ae3-tl-build/README.md) and [`ae3-tl-build/docs/API.md`](ae3-tl-build/docs/API.md).

## Requirements

- An x86_64 Linux host with an NVIDIA GPU and driver **550 or newer** (tested on 580).
  Check with `nvidia-smi`.
- Docker with the **NVIDIA Container Toolkit**, so `docker run --gpus` works.
- About **3 GB of GPU memory per model copy** under load (see [Sizing](#sizing)).

---

## 1. Run it

```bash
# A key that stays on the server: never paste it into chats, tickets or code
openssl rand -hex 24 > ~/ae3-tl.key && chmod 600 ~/ae3-tl.key

docker run -d --name ae3-tl --restart unless-stopped \
  --gpus all -p 8000:8000 --shm-size 1g \
  -e AE3_API_KEY=$(cat ~/ae3-tl.key) \
  -e AE3_COPIES=3 \
  -v ae3tl-weights:/data/weights \
  apexe3/ae3-table-extract:gpu

docker logs -f ae3-tl
```

On first start the weights (about 126 MB) download into the `ae3tl-weights` volume. Then the
entrypoint opens the model on CUDA and prints **`GPU check passed`**, and each copy prints
`Application startup complete` after about a minute of warm-up.

> The container is started with `docker run`, so read its logs with `docker logs`, not
> `docker compose logs` (that needs a compose file and answers "no configuration file provided").

## 2. Check that it really runs on the GPU

ONNX Runtime falls back to the CPU **silently** when CUDA cannot load, and the same workload then
runs about 35 times slower with no error. The entrypoint refuses to start in that case and says why.
To confirm by hand:

```bash
curl localhost:8000/ready
nvidia-smi
```

`nvidia-smi` should list one `/opt/venv/bin/python` process per copy on the GPU, each holding
0.7–3 GB. To watch the GPU work, run `nvidia-smi dmon -s u` while sending load: the `sm` column
rises during a burst and falls back to 0 between requests. A single page uses the GPU for only a
few tens of milliseconds, so one request at a time can show 0%; that is normal.

## 3. Send requests the fast way

The benchmark speed assumes the page's **words are attached** and nothing else runs on the CPU:

```bash
curl -s localhost:8000/v1/extract -H "X-API-Key: $(cat ~/ae3-tl.key)" \
  -F "file=@page.png" \
  -F "words=@page_words.json" \
  -F "ocr=never" \
  -F "rotation=0" \
  -F "format=json"
```

| field | why |
|---|---|
| `words` | the page's words with their boxes (JSON, XML or CSV; see the API docs). For a digital PDF, take them from its text layer: exact text, no OCR errors. |
| `ocr=never` | never run OCR for this request |
| `rotation=0` | skip the automatic rotation check (it runs Tesseract) |
| `format` | `json`, `markdown`, `html` or `csv` |

### Getting the words

A words file is a list of the page's words with their boxes, in the pixels of the image you send:

```json
[{"text": "Characteristics", "bbox": [158, 163, 278, 176], "block_num": 1001, "line_num": 1, "span_num": 1}, ...]
```

`scripts/make_words.py` makes them with the service's own code, so the boxes always line up:

| your documents | what it does | speed |
|---|---|---|
| PDFs with a text layer (most digital documents) | renders each page to PNG at 200 DPI and reads the words from the PDF itself: exact text, no OCR | fast |
| scans, photos, image-only PDFs | reads the words with Tesseract (this is how `ae3-tl-build/examples/sample_words.json` was made) | seconds a page, CPU |

Run it inside the CPU image, which already has the service code and Tesseract. From this folder:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work \
  --entrypoint python apexe3/ae3-table-extract:cpu \
  scripts/make_words.py report.pdf scans/ --out pages/
```

Each page becomes `pages/<name>.png` plus `pages/<name>.json`. Add `--ocr-empty` to read
image-only pages inside a PDF with Tesseract. Any other OCR (PaddleOCR, Textract, Azure, Google
Vision) works too, as long as its boxes are in the image's pixels; JSON, XML and CSV are all
accepted (see [`ae3-tl-build/docs/API.md`](ae3-tl-build/docs/API.md)).

Words never come from the GPU itself, so for scans the CPU work moves to this step. Run it
ahead of time, or on other machines, to keep the GPU service at full speed.

### Try it with the sample page

`ae3-tl-build/examples/` has a real page and its words. On the server, from this folder:

```bash
curl -s localhost:8000/v1/extract -H "X-API-Key: $(cat ~/ae3-tl.key)" \
  -F "file=@ae3-tl-build/examples/sample.png" -F "words=@ae3-tl-build/examples/sample_words.json" \
  -F "ocr=never" -F "rotation=0" -F "format=markdown"
```

### Load test from another machine

From your laptop (its address must be allowed through the server's firewall), in this
folder's `ae3-tl-build/examples/` (or the repository's `examples/`), with the key in `AE3_KEY`. Keep it **on one line**:

```bash
seq 200 | xargs -P 16 -I{} curl -s -o /dev/null -w "%{http_code}\n" http://<server-ip>:8000/v1/extract -H "X-API-Key: $AE3_KEY" -F "file=@sample.png" -F "words=@sample_words.json" -F "ocr=never" -F "rotation=0" | sort | uniq -c
```

`200 200` means every page succeeded. Replace `%{http_code}\n` with `%{http_code} %{time_total}\n`
and drop `| sort | uniq -c` to see each request's time.

The GPU image contains **no Tesseract**: a page sent without words comes back with the table
structure but no cell text. For scanned pages, run the CPU image, which reads them with Tesseract
on the CPU (seconds per page instead of milliseconds).

The response's `elapsed_ms` is the service's own processing time. Anything above that in your
client's timing is upload and network: from a laptop on a home connection a 400 KB page can take
seconds to arrive, while the same request from a server in the same cloud region takes
milliseconds.

## 4. Sizing

`AE3_COPIES` starts that many worker processes behind port 8000. Each has its own model on the GPU
and works on up to 8 pages at once (`AE3_CONCURRENCY`); the entrypoint splits the CPU cores between
them (`AE3_THREADS`).

| copies | GPU memory | RTX 5090, PubTables-v2 pages | A100 shared with vLLM, sample page with words |
|---|---|---|---|
| 1 | ~2.9 GB | 29.1 pages/s | |
| 3 | ~8.7 GB | | 37.7 pages/s |
| 4 | ~11.4 GB | 46.9 pages/s | |
| 8 | ~24.8 GB | 70.0 pages/s | |

### Sharing a GPU with another model

A vLLM model takes `--gpu-memory-utilization` of each GPU it uses (0.9 = 90%). Leave room for the
AE3-TL copies, start vLLM first (it measures free memory when it starts), then AE3-TL:

```bash
# vLLM on both GPUs, 85% of each: leaves ~12 GB per 80 GB A100
docker run -d --name minicpm --restart unless-stopped --gpus all --shm-size 1g \
  -p 127.0.0.1:8081:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:nightly <model> --tensor-parallel-size 2 --gpu-memory-utilization 0.85

# AE3-TL on the second GPU only, 3 copies (~8.7 GB)
docker run -d --name ae3-tl --restart unless-stopped --gpus '"device=1"' -p 8000:8000 --shm-size 1g \
  -e AE3_API_KEY=$(cat ~/ae3-tl.key) -e AE3_COPIES=3 \
  -v ae3tl-weights:/data/weights apexe3/ae3-table-extract:gpu
```

Inside the container the chosen GPU appears as GPU 0, so no other setting is needed. For 4 copies,
lower vLLM to `0.8`. Both share the GPU's compute, so each slows a little when both are busy.

## 5. Measure it

```bash
export AE3_API_KEY=$(cat ~/ae3-tl.key)

# 300 pages from 24 clients, on the server itself, from this folder
python3 scripts/bench.py --images ae3-tl-build/examples/sample.png \
  --words ae3-tl-build/examples/sample_words.json --count 300 --concurrency 24

# your own pages, made by scripts/make_words.py (page.png <-> page.json)
python3 scripts/bench.py --images pages/ --words pages/ --concurrency 24
```

It prints every HTTP status, pages per second and per hour, and client and server latency, and
exits non-zero unless every page succeeded. Run it on the server to measure the service; run it
from elsewhere to include the network. Quote numbers from a mix of real pages, not one page repeated.

## 6. Security

- **The key:** keep it in `~/ae3-tl.key` on the server, read it with `$(cat ~/ae3-tl.key)`. A key
  that has been pasted anywhere else is exposed: replace it (new key file, then remove and re-run
  the container; the weights are kept in the volume).
- **Reachability:** `-p 8000:8000` listens on every interface. Either keep the cloud firewall to an
  allowlist of addresses, or listen on the server only with `-p 127.0.0.1:8000:8000` and put HTTPS
  in front, for example Caddy with automatic certificates:

  ```bash
  mkdir -p ~/caddy && cat > ~/caddy/Caddyfile <<'EOF'
  tables.example.com {
      request_body {
          max_size 64MB
      }
      reverse_proxy 127.0.0.1:8000
  }
  EOF
  docker run -d --name caddy --restart unless-stopped --network host \
    -v ~/caddy/Caddyfile:/etc/caddy/Caddyfile -v caddy_data:/data caddy:2
  ```

  Open ports 80 and 443 only. `/health`, `/ready` and `/metrics` answer without a key.
- **Other services on the same host:** a vLLM server has no key of its own; publish it on
  `127.0.0.1` unless something outside must reach it.

## 7. Build and publish the images

Build on an **x86_64 Linux server**. On an Apple-silicon Mac the GPU image builds under emulation
and the step that installs about 2 GB of NVIDIA libraries can take over half an hour.

Copy this folder to the server as one archive, replacing any older copy so no deleted files
linger (Google Cloud example; run from the repository root on your own machine):

```bash
tar --exclude='__pycache__' --exclude='.DS_Store' -czf /tmp/ae3-gpu-setup.tgz ae3-gpu-setup
gcloud compute scp /tmp/ae3-gpu-setup.tgz <instance>:~/ --project=<project> --zone=<zone>
gcloud compute ssh <instance> --project=<project> --zone=<zone>
```

Then on the server:

```bash
cd ~
[ -d ae3-gpu-setup ] && mv ae3-gpu-setup ae3-gpu-setup.old-$(date +%Y%m%d)
tar -xzf ae3-gpu-setup.tgz && rm ae3-gpu-setup.tgz
cd ~/ae3-gpu-setup
docker login -u <your Docker ID>     # a personal access token, not your password
./build_and_push.sh                  # :gpu, :0.24-gpu, :cpu, :0.24-cpu
docker logout
```

To move a running service to the new image (the weights stay in the volume):

```bash
docker rm -f ae3-tl
docker run -d --name ae3-tl --restart unless-stopped --gpus '"device=1"' -p 8000:8000 --shm-size 1g \
  -e AE3_API_KEY=$(cat ~/ae3-tl.key) -e AE3_COPIES=3 \
  -v ae3tl-weights:/data/weights apexe3/ae3-table-extract:0.24-gpu
```

- Log in as **your own Docker ID**. An organisation such as `apexe3` cannot log in; your account
  pushes to the organisation's repository when it is a member of a team with Read & Write access.
  Create the token under Account settings → Personal access tokens (Read & Write).
- `build_and_push.sh` builds the CPU image for x86_64 and ARM. On an x86 server it installs QEMU
  first; `CPU_PLATFORMS=linux/amd64 ./build_and_push.sh` skips ARM. `VERSION=0.25` changes the tag.
- To build only the GPU image:
  `docker build -t apexe3/ae3-table-extract:gpu -t apexe3/ae3-table-extract:0.24-gpu .`
  then `docker push` both tags.

## 8. Run without Docker

On a GPU machine with a CUDA 12 runtime (for example a RunPod or cloud GPU image):

```bash
bash setup_gpu.sh                                  # Python 3.11 + pinned packages, venv in /opt/ae3-venv
AE3_API_KEY=$(cat ~/ae3-tl.key) AE3_COPIES=3 bash serve.sh
```

`serve.sh` runs the same entrypoint as the image, including the GPU check.

## Settings

| variable | default | |
|---|---|---|
| `AE3_API_KEY` | none | required; the service refuses to start without it |
| `AE3_COPIES` | 1 | model copies (worker processes) |
| `AE3_CONCURRENCY` | min(8, cores) | pages in flight per copy |
| `AE3_THREADS` | cores / copies | ONNX Runtime CPU threads per copy |
| `AE3_BATCH_SIZE` | 1 | keep at 1: batch 16 gave identical output but was 13% slower |
| `AE3_PROVIDER` | `cuda` (GPU image), `cpu` (CPU image) | |
| `OCR_ENGINE` | `none` (GPU image), `tesseract` (CPU image) | |
| `AE3_TF32` | off | `1` is faster on A100-class GPUs; results can shift slightly between runs |
| `AE3_PDF_INPUT` | 0 | `1` accepts PDFs and reads their text layer; not benchmarked yet |
| `AE3_WEIGHTS_DIR` | `/data/weights` | mount a volume here so the weights download once |
| `AE3_SKIP_GPU_CHECK` | 0 | `1` starts even when CUDA is not in use (testing only) |

More limits and options: [`ae3-tl-build/.env.example`](ae3-tl-build/.env.example).

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `CUDA is NOT in use` and the container stops | start it with `--gpus`; install the NVIDIA Container Toolkit; driver 550+ |
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | an old container still holds the port: `docker ps --filter publish=8000`, then `docker stop` and `docker rm` it |
| `401 unauthorized` after re-creating the container | each `docker run` with `$(openssl rand ...)` makes a new key; use the key file, or read the live key with `docker inspect ae3-tl --format '{{range .Config.Env}}{{println .}}{{end}}' \| grep AE3_API_KEY` |
| tables come back with empty cells | no words were sent to the GPU image; make them with `scripts/make_words.py` and attach `words` |
| `400 invalid_words` or text in the wrong cells | the words' boxes are not in the pixels of the image sent; make the image and its words together with `scripts/make_words.py` |
| seconds per request from a laptop, but `elapsed_ms` is ~200 | upload time over the internet, not the service; measure on the server with `scripts/bench.py` |
| GPU shows 0% in `nvidia-smi dmon` | one request at a time keeps the GPU busy for only ~20–40 ms a page; send parallel load |
| requests were fast, then seconds each | check the request has `rotation=0` and words; a rotation check or OCR runs on the CPU |
| `404 file_not_found` for a `file_id` | with more than one copy, uploads are cached per copy; send the image with every request |
| `GET /v1/extract 405 Method Not Allowed` in the log | the request arrived with no file: usually a space after a trailing `\` broke the command in two. Put it on one line, or delete the spaces after each `\` |
| zsh `parse error near '\n'` when pasting commands | a placeholder such as `<key>` was left in; replace it, angle brackets included |
