# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import sys
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch, os

# Make the retrofit compile_utils helper importable from the production server.
# Path mirrors the one used by retrofit/bench scripts.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "retrofit")
)
try:
    from compile_utils import wrap_compile  # type: ignore
except Exception as _e:  # noqa: BLE001 — fall back to a passthrough on any failure
    logging.warning("retrofit/compile_utils unavailable (%s); compile disabled.", _e)
    def wrap_compile(model, mode=None, dynamic=None, label="model"):  # type: ignore
        return model


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    # Skip config (enable_skipping / dynamic_skip_config / use_cache) is now
    # threaded per-request from the client via vla_input; no server-side
    # pre-configuration of the adapter needed.
    vla = vla.to("cuda").eval()
    # Iso-cost paper claim: wrap the VLA policy with torch.compile so the
    # default LIBERO inference matches the speed-bench number cited in the
    # paper. Pass --compile-mode off to bypass for accuracy reproduction.
    vla = wrap_compile(vla, mode=args.compile_mode, label="server_policy.vla")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--compile-mode",
        default=None,
        help=(
            "torch.compile mode for the policy at serve time. None → "
            "RETROFIT_COMPILE_MODE env or 'max-autotune-no-cudagraphs' (paper "
            "iso-cost default). Pass 'off' to disable compile for accuracy "
            "reproduction."
        ),
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
