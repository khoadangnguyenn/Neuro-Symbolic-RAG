import urllib.request
import json
import random
import sys

def test_api():
    print("\n" + "="*40)
    print("🚀 EXACT PIPELINE INTERACTIVE TESTER")
    print("="*40)
    print("1. Lấy câu hỏi Logic ngẫu nhiên từ Dataset (Type 2)")
    print("2. Lấy câu hỏi Vật lý ngẫu nhiên từ Dataset (Type 2)")
    print("3. Tải câu hỏi Toán đố (Type 1) ngẫu nhiên từ nguồn ngoài (URL/Local File)")
    print("4. Tự nhập câu hỏi bằng tay (Type 1 hoặc 2)")
    print("5. Thoát")
    
    choice = input("\n👉 Chọn tính năng (1/2/3/4/5): ").strip()
    
    if choice == '5':
        sys.exit(0)
        
    query_type = "type2"
    question = ""
    premises = None
    
    if choice == '1':
        try:
            with open("dataset/Logic_Based_Educational_Queries.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                item = random.choice(data)
                question = random.choice(item["questions"])
                premises = item.get("premises-NL", [])
                query_type = "type2"
                print(f"\n[?] Câu hỏi: {question}")
                print(f"[*] Premises (giả định): {len(premises)} câu")
        except Exception as e:
            print(f"Lỗi đọc file: {e}")
            return
            
    elif choice == '2':
        try:
            with open("dataset/Physics_Calculation_Problems.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                item = random.choice(data)
                question = item["question"]
                query_type = "type2"
                print(f"\n[?] Câu hỏi: {question}")
        except Exception as e:
            print(f"Lỗi đọc file: {e}")
            return
            
    elif choice == '3':
        print("\n[*] Bạn có thể dán đường dẫn URL hoặc đường dẫn file Local.")
        print("[*] Hỗ trợ định dạng: .json (Mảng các object) hoặc .jsonl (Mỗi dòng 1 object).")
        print("💡 Ví dụ URL file JSON: https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl")
        source = input("\n[?] Nhập URL hoặc đường dẫn file: ").strip()
        
        try:
            print("Đang tải dữ liệu...")
            data_text = ""
            if source.startswith("http://") or source.startswith("https://"):
                req_ext = urllib.request.Request(source, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req_ext) as resp:
                    data_text = resp.read().decode('utf-8')
            else:
                with open(source, "r", encoding="utf-8") as f:
                    data_text = f.read()

            # Xử lý parse json hoặc jsonl
            data_items = []
            if source.endswith(".jsonl") or "\n{" in data_text:
                for line in data_text.strip().split("\n"):
                    if line.strip():
                        data_items.append(json.loads(line))
            else:
                data_items = json.loads(data_text)
                
            item = random.choice(data_items)
            # Thử các trường phổ biến chứa câu hỏi
            question = item.get("question") or item.get("instruction") or item.get("q") or list(item.values())[0]
            query_type = "type1"
            
            print(f"\n[?] Đã bốc ngẫu nhiên 1 câu Type 1 từ {len(data_items)} câu.")
            print(f"[?] Câu hỏi: {question}")
            
        except Exception as e:
            print(f"Lỗi tải/đọc file: {e}")
            return

    elif choice == '4':
        question = input("\n[?] Nhập câu hỏi của bạn: ")
        query_type = input("[?] Query Type (type1/type2) [mặc định type1]: ").strip() or "type1"
        if query_type == "type2":
            print("[*] (Tùy chọn) Nhập premises cho type2. Bỏ qua thì nhấn Enter.")
            prem_input = input("Premise: ").strip()
            if prem_input:
                premises = [prem_input]
    else:
        print("Lựa chọn không hợp lệ!")
        return

    # Chuẩn bị payload
    payload = {
        "query_type": query_type,
        "question": question
    }
    if premises is not None:
        payload["premises-NL"] = premises

    req = urllib.request.Request(
        "http://localhost:8000/answer",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    print("\n⏳ Đang xử lý... (Có thể mất 10-30s nếu hệ thống gọi Llama.cpp để code)")
    
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode('utf-8'))
            print("\n" + "="*40)
            print("✅ KẾT QUẢ TRẢ VỀ:")
            print("="*40)
            print(f"🎯 Answer: {result.get('answer')}")
            print(f"🤖 Confidence: {result.get('confidence')}")
            print("📝 Explanation:", result.get("explanation", "N/A"))
            
            # --- In lỗi thực thi Python (nếu có) ---
            metadata = result.get("metadata", {})
            if "execution_errors" in metadata:
                print("\n❌ LỖI PYTHON SANDBOX:")
                for err in metadata["execution_errors"]:
                    print(f"  - {err}")
            elif metadata.get("executor") == "python" and result.get("source") != "self-hosted-llm-fallback":
                print(f"\n💻 Python Sandbox: Execution SUCCESS")
            
            print("\n🧠 Chain of Thought (CoT):")
            for step in result.get('cot', []):
                print(f"  - {step}")
    except Exception as e:
        print(f"\n❌ Lỗi kết nối tới API: {e}")
        print("💡 Gợi ý: Kiểm tra xem 'docker compose up' đã chạy xong và báo 'Running on http://0.0.0.0:8000' chưa.")

if __name__ == "__main__":
    while True:
        test_api()
        if input("\n🔄 Tiếp tục test? (y/n): ").lower() != 'y':
            break
