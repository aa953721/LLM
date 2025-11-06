import os
import sys
from pathlib import Path
from datetime import datetime

from google import genai


MODEL_TEXT = "gemini-2.5-flash"
MODEL_IMAGE = "imagen-3.0-generate-002"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


def require_api_key() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY is not set. Set it in this PowerShell session and try again.")
        print("Example: $env:GEMINI_API_KEY = \"YOUR_API_KEY\"")
        sys.exit(2)


def client() -> genai.Client:
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def chat(message: str) -> int:
    require_api_key()
    resp = client().models.generate_content(model=MODEL_TEXT, contents=message)
    print(getattr(resp, "text", "(no response)"))
    return 0


def analyze_image(path: str) -> int:
    require_api_key()
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}")
        return 1
    cl = client()
    uploaded = cl.files.upload(path=str(p))
    resp = cl.models.generate_content(
        model=MODEL_TEXT,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "Describe this image and list notable objects."},
                    {"file_data": {"file_uri": uploaded.uri, "mime_type": uploaded.mime_type}},
                ],
            }
        ],
    )
    print(getattr(resp, "text", "(no response)"))
    return 0


def generate_image(prompt: str) -> int:
    require_api_key()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    cl = client()
    try:
        resp = cl.models.generate_content(
            model=MODEL_IMAGE,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
    except Exception as e:
        print(f"Image generation request failed: {e}")
        return 1

    images: list[bytes] = []
    try:
        parts = []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if content and getattr(content, "parts", None):
                parts.extend(content.parts)
        for p in parts:
            d = getattr(p, "inline_data", None) or getattr(p, "inlineData", None)
            if d and str(getattr(d, "mime_type", "")).startswith("image/"):
                data = getattr(d, "data", None)
                if isinstance(data, (bytes, bytearray)):
                    images.append(bytes(data))
                elif isinstance(data, str):
                    import base64
                    images.append(base64.b64decode(data))
    except Exception:
        pass


    if not images:
        data = getattr(resp, "image", None)
        if data:
            images = [data]

    if not images:
        print("No image returned. Check API key, model access, and quota.")
        return 1

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"gen-{ts}.png"
    out_path.write_bytes(images[0])
    print(f"Saved: {out_path}")
    return 0


def print_usage() -> None:
    print(
        "\n".join(
            [
                "Usage:",
                "  python simple_assistant.py chat <message>",
                "  python simple_assistant.py analyze-image <path>",
                "  python simple_assistant.py generate-image <prompt>",
                "  python simple_assistant.py help",
            ]
        )
    )



def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help", "help"}:
        print_usage()
        return 1 if len(argv) < 2 else 0

    cmd = argv[1]
    args = argv[2:]
    try:
        if cmd == "chat":
            if not args:
                print("Provide a message.")
                return 1
            return chat(" ".join(args))
        if cmd == "analyze-image":
            if not args:
                print("Usage: python simple_assistant.py analyze-image <path>")
                return 1
            return analyze_image(args[0])
        if cmd == "generate-image":
            if not args:
                print("Usage: python simple_assistant.py generate-image <prompt>")
                return 1
            return generate_image(" ".join(args))
        print("Unknown command. Try: python simple_assistant.py help")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv))
