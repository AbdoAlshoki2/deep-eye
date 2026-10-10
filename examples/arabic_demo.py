"""Writes a small trace with Arabic input and output, to try the viewer's right-to-left mode.

    python examples/arabic_demo.py
    deep-eye            # press `b` to toggle right-to-left (needs: pip install "deep-eye[rtl]")
"""
import deep_eye


@deep_eye.trace(kind="tool")
def translate(text: str) -> str:
    return "Hello, how are you today?"


@deep_eye.trace(kind="llm")
def answer(question: str) -> dict:
    return {
        "reply": "مرحبا! أنا بخير، شكرا لسؤالك. كيف يمكنني مساعدتك اليوم؟",
        "note": "Mixed text: الذكاء الاصطناعي (AI) يساعد في كتابة الكود.",
    }


@deep_eye.trace(name="agent")
def run(question: str) -> str:
    translate(question)
    return answer(question)["reply"]


if __name__ == "__main__":
    print(ascii(run("مرحبا، كيف حالك اليوم؟")))
