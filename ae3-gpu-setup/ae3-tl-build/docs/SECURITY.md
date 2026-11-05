# Security

- One API key per client environment, in `AE3_API_KEY`, sent as `X-API-Key` or a Bearer token. The
  service refuses to start without it, and every `/v1` and `/mcp` path answers `401` without it.
  The key is compared in constant time with `hmac.compare_digest`. `/health`, `/ready` and
  `/metrics` stay open for probes and scrapers.
- Limits: `AE3_MAX_IMAGE_MB` (25), `AE3_MAX_IMAGE_PIXELS` (60M), `AE3_MAX_PDF_MB` (50),
  `AE3_MAX_PAGES` (100), `AE3_QUEUE_LIMIT` (8 running extractions), `AE3_JOB_LIMIT` (8 jobs). A full
  queue answers 429 with Retry-After.
- PDF input is off in this release (`AE3_PDF_INPUT=0`). A PDF is refused with `415` before it is
  parsed.
- Outbound is only `file_url` and `words_url`, and only http(s). It is off by default:
  `AE3_ALLOW_OUTBOUND=0`. With `1`, the service also needs `AE3_FETCH_ALLOWLIST`, a comma list of
  hosts, and it refuses to start without it. A host is allowed when it matches an entry exactly or
  when it ends with an entry written as `.example.com`. The allowlist is passed to the fetch as a
  keyword argument, never by position.
- Every fetch is checked hop by hop. The scheme is http or https, the host must be on the
  allowlist, and then the name is resolved. Every address it resolves to must be public: private
  (10/8, 172.16/12, 192.168/16), loopback (127/8, ::1), link-local (169.254/16, fe80::/10),
  multicast, reserved and cloud metadata addresses (169.254.169.254, metadata.google.internal,
  fd00:ec2::254) are refused with `403`. A host written as one decimal number, such as
  `http://2130706433/`, is read as the address it stands for, so it cannot slip past the check.
  Redirects are followed one hop at a time and every hop is checked again; a redirect into a
  private address is refused. The connection is made to the address that was checked, with the
  original host in the Host header and as the TLS server name, so DNS cannot be re-pointed between
  the check and the request.
- Words files are XML, JSON or CSV. The XML is parsed with `defusedxml`, so an entity expansion or
  an external reference is refused instead of resolved.
- The container runs as a non-root user, with a read-only root filesystem, `/tmp` on tmpfs, all
  capabilities dropped and `no-new-privileges`.
- No request writes to disk. Uploads live in a bounded in-process cache and jobs live in memory for
  `AE3_JOB_TTL_S`.
- Logs carry counts and timings only; table text, file names and ids are not logged.

## State in memory, and what it means for more than one copy

The service is not stateless. `file_id`, jobs and crop all live in the memory of the copy that
answered the request. One copy therefore answers a `file_id` twice, keeps a job pollable, and can
crop from a recent upload. Several copies (`AE3_COPIES` above 1) do not share that memory.

In scaled mode: send the image or the `file_url` with every request, ask for `include_crops=true`
when you need crops, and keep a job on the copy that started it. Do not rely on `file_id`, on the
crop URL, or on `GET /v1/jobs/{id}` landing on the same copy. The 2-copy behaviour is measured in
`docs/SIZING.md`.
