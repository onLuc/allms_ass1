import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_cleanup, compute_init
from nanochat.engine import Engine


PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nanochat experiment</title>
<style>body{font:16px system-ui;max-width:760px;margin:3rem auto;padding:0 1rem;color:#17212b}textarea{width:100%;min-height:100px}button,select{padding:.6rem;margin:.5rem 0}pre{white-space:pre-wrap;background:#f3f5f7;padding:1rem;min-height:5rem}</style>
<h1>Nanochat experiment</h1><label for="prompt">Message</label><textarea id="prompt"></textarea>
<label for="temperature">Temperature</label><select id="temperature"><option>0.1</option><option selected>0.7</option><option>1.5</option></select>
<button id="send">Generate</button><pre id="answer"></pre>
<script>document.querySelector('#send').onclick=async()=>{const b=document.querySelector('#send'),o=document.querySelector('#answer');b.disabled=true;o.textContent='Generating…';try{const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:document.querySelector('#prompt').value,temperature:Number(document.querySelector('#temperature').value)})});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);o.textContent=d.response}catch(e){o.textContent=String(e)}finally{b.disabled=false}}</script>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["base", "mid", "sft"], default="sft")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    _, _, _, _, device = compute_init(args.device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    max_context = meta["model_config"]["sequence_len"]

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, value):
            content = json.dumps(value).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 32000:
                self.send_json(413, {"error": "Request must be between 1 and 32,000 bytes"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                prompt = body.get("prompt", "")
                temperature = float(body.get("temperature", 0.7))
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
                    raise ValueError("Enter a prompt of 1 to 12,000 characters")
                if not math.isfinite(temperature) or not 0 <= temperature <= 2:
                    raise ValueError("Temperature must be between 0 and 2")
                prompt_ids = tokenizer.encode(prompt)
                if len(prompt_ids) + 132 > max_context:
                    raise ValueError(f"Prompt is too long for this model's {max_context}-token context")
                ids = [bos, user_start, *prompt_ids, user_end, assistant_start]
                generated = []
                for column, _ in engine.generate(ids, max_tokens=128, temperature=temperature, top_k=50):
                    token = int(column[0].item())
                    if token == assistant_end:
                        break
                    generated.append(token)
                self.send_json(200, {"response": tokenizer.decode(generated), "temperature": temperature, "model_tag": args.model_tag, "checkpoint_step": meta["step"]})
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                self.send_json(400, {"error": str(error)})

        def log_message(self, format_string, *values):
            print(f"{self.address_string()} - {format_string % values}")

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Model {args.source}/{args.model_tag} step {meta['step']} listening at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        compute_cleanup()


if __name__ == "__main__":
    main()
