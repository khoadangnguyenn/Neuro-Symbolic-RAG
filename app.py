import streamlit as st
import requests
import time

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="EXACT Neuro-Symbolic AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. MODERN CUSTOM CSS
st.markdown("""
<style>
    /* Gradient Text for Main Header */
    .gradient-text {
        background: linear-gradient(45deg, #1e3a8a, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-color);
        opacity: 0.7;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: -10px;
        margin-bottom: 40px;
        letter-spacing: 0.5px;
    }

    /* Style for Result Box - Light/Dark Mode Compatible */
    .result-card {
        background: rgba(40, 167, 69, 0.05);
        border-left: 6px solid #28a745;
        border-radius: 10px;
        padding: 25px;
        margin-top: 15px;
        margin-bottom: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.08);
    }
    
    .answer-highlight {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-color);
        display: block;
        margin-top: 10px;
    }

    /* Override Streamlit Button */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 0;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    [data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
    }
    
    /* Custom Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(150,150,150,0.2), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 3. SIDEBAR (Clean and modern layout)
with st.sidebar:
    st.markdown("<div style='text-align: center;'><span style='font-size: 4rem;'>🧠</span></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>EXACT AI</h2>", unsafe_allow_html=True)
    st.caption("<p style='text-align: center;'>HybridDB & LLM Architecture</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### 📖 Operational Pipeline")
    st.info("""
    1. **Classification**: Physics / Math vector space.
    2. **Querying**: GraphDB analysis.
    3. **Execution**: Code generation via Python Sandbox.
    """, icon="⚡")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.success("👨‍💻 **Project: XAI Competition 2026**\n\nStatus: Ready")

# 4. MAIN HEADER
st.markdown("<h1 class='gradient-text'>EXACT Neuro-Symbolic</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Explainable AI (XAI) Solution integrating Knowledge Graphs & Python Sandbox</p>", unsafe_allow_html=True)

# 5. INPUT SECTION (Card-like interface)
with st.container():
    st.markdown("### 🎯 Input Parameters")
    
    col1, col2 = st.columns([1, 2.5], gap="large")
    with col1:
        query_type = st.selectbox(
            "📚 Domain:",
            ["Physics", "Math & Logic"],
            index=0,
            help="The system will automatically select the appropriate Vector space to optimize accuracy."
        )
        q_type_str = "type2" if "Physics" in query_type else "type1"

    with col2:
        question = st.text_area(
            "📝 Question Content:",
            height=120,
            placeholder="Example: Calculate the energy stored in a capacitor C when C = 100 μF and U = 30 V..."
        )

# Submit Button
submit = st.button("✨ Activate Analysis System", type="primary", use_container_width=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# 6. PROCESSING & OUTPUT SECTION
if submit:
    if not question.strip():
        st.toast("⚠️ Please enter your question!", icon="⚠️")
        st.warning("You must provide a problem statement before running the system.")
    else:
        # Using st.status instead of st.spinner to create a step-by-step "AI thinking" feel
        with st.status("🧠 Initializing Neuro-Symbolic reasoning pipeline...", expanded=True) as status:
            try:
                start_time = time.time()
                
                # Mock steps to make the UI look more professional
                st.write("📥 Normalizing question semantics...")
                time.sleep(0.5) 
                st.write(f"🔍 Retrieving Knowledge Graph for the **{query_type.split(' ')[0]}** domain...")
                
                payload = {
                    "query_type": q_type_str,
                    "question": question
                }
                
                import json
                try:
                    # Attempt to parse as JSON in case user pastes raw JSON format
                    json_text = question.strip()
                    if not json_text.startswith("{"):
                        json_text = "{" + json_text + "}"
                    parsed_json = json.loads(json_text)
                    if isinstance(parsed_json, dict):
                        payload.update(parsed_json)
                except Exception:
                    pass
                
                # Request to internal Docker API
                response = requests.post("http://localhost:8000/answer", json=payload, timeout=300)
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.write("⚙️ Executing Python Sandbox & Synthesizing results...")
                    time.sleep(0.5)
                    
                    status.update(label=f"✅ Analysis completed in {end_time - start_time:.2f} seconds!", state="complete", expanded=False)
                    
                    # 7. PREMIUM RESULT DISPLAY
                    st.success(f"✅ Processing completed in **{end_time - start_time:.2f} seconds**!")
                    
                    # Metadata Column
                    source = data.get("source", "N/A")
                    confidence = data.get("confidence", 0.0)
                    st.markdown(f"**Execution Source:** `{source}` | **Confidence Score:** `{confidence * 100:.1f}%`")
                    
                    # Output Answer Box
                    answer_text = data.get("answer", "No specific answer found.")
                    st.markdown(f"""
                    <div class="result-box">
                        <h4 style="margin-top:0; color:#28a745; font-weight: bold;">💡 System Answer:</h4>
                        <span style="font-size: 1.3em; font-weight: 500;">{answer_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Explanation
                    explanation_text = data.get("explanation", "")
                    if explanation_text:
                        st.markdown("### 📝 Explanation")
                        st.info(explanation_text)
                    
                    # Chain of Thought Expander
                    cot = data.get("cot", [])
                    fol = data.get("fol", "")
                    premises = data.get("premises", [])
                    
                    with st.expander("🔍 View detailed reasoning (Chain of Thought)", expanded=False):
                        if cot:
                            st.markdown("**Reasoning Steps (CoT):**")
                            cot_formatted = "\n".join([f"- {step}" for step in cot])
                            st.markdown(f"<div class='rationale-box'>{cot_formatted}</div>", unsafe_allow_html=True)
                        
                        if premises:
                            st.markdown("**Premises:**")
                            premises_formatted = "\n".join([f"- {p}" for p in premises])
                            st.markdown(f"<div class='rationale-box'>{premises_formatted}</div>", unsafe_allow_html=True)
                        
                        if fol:
                            st.markdown("**First-Order Logic (FOL):**")
                            st.markdown(f"<div class='rationale-box'>{fol}</div>", unsafe_allow_html=True)
                            
                        if not cot and not premises and not fol:
                            st.markdown("<div class='rationale-box'>No detailed reasoning steps available.</div>", unsafe_allow_html=True)
                            
                    with st.expander("📦 Full JSON Response (For Debugging)"):
                        st.json(data)
                        
                else:
                    status.update(label="❌ An error occurred during analysis!", state="error", expanded=False)
                    st.error(f"❌ API Server Error: HTTP {response.status_code}")
                    with st.expander("Error Details"):
                        st.write(response.text)
                        
            except requests.exceptions.ConnectionError:
                status.update(label="🔌 System connection lost!", state="error", expanded=False)
                st.error("**Cannot connect to the processing server!**")
                st.info("💡 Guide: Ensure you have started the backend using the command `docker-compose up -d` and the server is listening on port `8000`.")
            except requests.exceptions.Timeout:
                status.update(label="⏳ Response timeout!", state="error", expanded=False)
                st.error("**The problem is too complex or the server is under heavy load (Timeout).**")
            except Exception as e:
                status.update(label="⚠️ Unknown system error!", state="error", expanded=False)
                st.error(f"Error details: {str(e)}")