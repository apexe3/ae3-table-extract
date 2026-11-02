# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This software is free and open source: you may use, modify and distribute
# it, including in commercial products, under the terms of the License.
# AE3-TL is created and maintained by Team APEX:E3.

"""The inference backend. ONNX Runtime only: nothing here imports torch."""


def create_engine(settings):
    """The engine for this deployment, loaded once per process.

    Weights are fetched from the model repository first when they are not on disk:
    the image ships without them.
    """
    from ae3_tl.backends.engine_ort import EngineOrt
    from ae3_tl.weights import ensure_weights

    ensure_weights(settings.weights_dir)
    return EngineOrt(settings)
