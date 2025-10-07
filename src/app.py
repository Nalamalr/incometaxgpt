"""
IncomeTaxGPT Web Interface using Streamlit
Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import our modules (adjust paths as needed)
# from src.embedding_system import EmbeddingSystem
# from src.retrieval_pipeline import RetrievalPipeline
# from src.llm_integration import IncomeTaxGPT, IncomeTaxGPT_API
# from src.tax_calculators import TaxCalculators

# Page config
st.set_page_config(
    page_title="IncomeTaxGPT - AI Tax Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .citation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .calculator-result {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

def initialize_system():
    """Initialize all components (cached)"""
    if not st.session_state.system_initialized:
        with st.spinner("🔄 Initializing IncomeTaxGPT..."):
            try:
                # Initialize components
                # st.session_state.embedding_sys = EmbeddingSystem()
                # st.session_state.embedding_sys.load_indices()
                # st.session_state.retrieval = RetrievalPipeline(st.session_state.embedding_sys)
                # st.session_state.calculators = TaxCalculators()
                
                # For demo, use API version
                # st.session_state.gpt = IncomeTaxGPT_API()
                
                st.session_state.system_initialized = True
                return True
            except Exception as e:
                st.error(f"❌ Initialization failed: {e}")
                return False
    return True

def display_header():
    """Display app header"""
    st.markdown('<h1 class="main-header">📊 IncomeTaxGPT</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.1rem; color: #666;'>
    Your AI-powered assistant for Indian Income Tax Law 🇮🇳
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

def display_sidebar():
    """Display sidebar with tools and settings"""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=TaxGPT", 
                use_column_width=True)
        
        st.markdown("### 🛠️ Tools & Features")
        
        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["💬 Chat Assistant", "🧮 Tax Calculators", "📚 Browse Sections"],
            index=0
        )
        
        st.markdown("---")
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.slider("Response Length", 100, 1000, 500, 50, key="response_length")
            st.slider("Number of Sources", 1, 10, 5, 1, key="num_sources")
            st.checkbox("Show Retrieved Context", value=False, key="show_context")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        if st.button("🗑️ Clear Chat"):
            st.session_state.conversation_history = []
            st.rerun()
        
        if st.button("💾 Export Chat"):
            export_conversation()
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 Session Stats")
        st.metric("Questions Asked", len(st.session_state.conversation_history) // 2)
        st.metric("System Status", "✅ Active" if st.session_state.system_initialized else "❌ Not Ready")
        
        return mode

def chat_interface():
    """Main chat interface"""
    st.markdown("### 💬 Ask Your Tax Question")
    
    # Example questions
    st.markdown("**Example questions:**")
    example_cols = st.columns(3)
    examples = [
        "What is Section 80C deduction limit?",
        "How to calculate HRA exemption?",
        "Compare old vs new tax regime"
    ]
    
    for col, example in zip(example_cols, examples):
        if col.button(example, key=f"ex_{example[:10]}"):
            st.session_state.current_query = example
    
    # Query input
    query = st.text_input(
        "Your Question:",
        value=st.session_state.get('current_query', ''),
        placeholder="E.g., What deductions are available under Section 80D?",
        key="query_input"
    )
    
    col1, col2 = st.columns([6, 1])
    with col1:
        submit = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        clear_query = st.button("🔄", use_container_width=True)
        if clear_query:
            st.session_state.current_query = ""
            st.rerun()
    
    # Process query
    if submit and query:
        with st.spinner("🤔 Thinking..."):
            process_query(query)
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 📜 Conversation History")
        
        for i, message in enumerate(reversed(st.session_state.conversation_history[-10:])):
            if message['role'] == 'user':
                st.markdown(f"**👤 You:** {message['content']}")
            else:
                with st.container():
                    st.markdown(f"**🤖 IncomeTaxGPT:**")
                    st.markdown(message['content'])
                    
                    # Show citations if available
                    if 'citations' in message and message['citations']:
                        with st.expander("📚 Sources & Citations"):
                            for citation in message['citations']:
                                st.markdown(f"- {citation}")
            
            st.markdown("---")

def process_query(query: str):
    """Process user query and generate response"""
    # Add to history
    st.session_state.conversation_history.append({
        'role': 'user',
        'content': query,
        'timestamp': datetime.now().isoformat()
    })
    
    # Demo response (replace with actual system)
    demo_response = f"""Based on the Income Tax Act, 1961:

**Answer to your question about: "{query}"**

This is a demo response. In the full system, this would be generated by:
1. Retrieving relevant sections from the tax corpus
2. Processing through the fine-tuned LLM
3. Generating a grounded response with citations

Example response structure:
- Clear explanation in plain English
- Reference to specific sections
- Calculation steps if applicable
- Practical guidance

[Section Reference: Example]

**Disclaimer:** This is informational guidance only and not professional tax advice. For complex situations, please consult a qualified Chartered Accountant.
"""
    
    # Add response to history
    st.session_state.conversation_history.append({
        'role': 'assistant',
        'content': demo_response,
        'citations': ['Section 80C, Income Tax Act 1961', 'Rule 2A, Income Tax Rules 1962'],
        'timestamp': datetime.now().isoformat()
    })
    
    st.rerun()

def calculator_interface():
    """Tax calculator tools interface"""
    st.markdown("### 🧮 Tax Calculators")
    
    calc_type = st.selectbox(
        "Select Calculator:",
        ["HRA Exemption", "Section 80C Deduction", "Tax Liability", "Regime Comparison"]
    )
    
    if calc_type == "HRA Exemption":
        st.markdown("#### House Rent Allowance (HRA) Exemption Calculator")
        
        col1, col2 = st.columns(2)
        with col1:
            basic = st.number_input("Annual Basic Salary (₹)", min_value=0.0, value=600000.0)
            hra_received = st.number_input("Annual HRA Received (₹)", min_value=0.0, value=240000.0)
        with col2:
            rent = st.number_input("Annual Rent Paid (₹)", min_value=0.0, value=180000.0)
            is_metro = st.checkbox("Living in Metro City", value=True)
        
        if st.button("Calculate HRA Exemption"):
            # Demo calculation
            metro_percent = 0.50 if is_metro else 0.40
            option1 = hra_received
            option2 = basic * metro_percent
            option3 = max(0, rent - (basic * 0.10))
            exemption = min(option1, option2, option3)
            
            st.markdown('<div class="calculator-result">', unsafe_allow_html=True)
            st.markdown("#### Calculation Result")
            st.metric("HRA Exemption", f"₹{exemption:,.2f}")
            st.metric("Taxable HRA", f"₹{hra_received - exemption:,.2f}")
            
            with st.expander("📊 Detailed Breakdown"):
                st.write(f"1. Actual HRA: ₹{option1:,.2f}")
                st.write(f"2. {int(metro_percent*100)}% of Basic: ₹{option2:,.2f}")
                st.write(f"3. Rent - 10% of Basic: ₹{option3:,.2f}")
                st.write(f"\n**Exemption = Minimum of above three: ₹{exemption:,.2f}**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif calc_type == "Section 80C Deduction":
        st.markdown("#### Section 80C Deduction Calculator")
        
        st.info("Enter your investments in various 80C eligible instruments:")
        
        col1, col2 = st.columns(2)
        with col1:
            ppf = st.number_input("PPF (₹)", min_value=0.0, value=50000.0)
            epf = st.number_input("EPF (₹)", min_value=0.0, value=40000.0)
            elss = st.number_input("ELSS (₹)", min_value=0.0, value=30000.0)
        with col2:
            life_insurance = st.number_input("Life Insurance Premium (₹)", min_value=0.0, value=25000.0)
            nsc = st.number_input("NSC (₹)", min_value=0.0, value=15000.0)
            other = st.number_input("Other 80C investments (₹)", min_value=0.0, value=0.0)
        
        if st.button("Calculate 80C Deduction"):
            total = ppf + epf + elss + life_insurance + nsc + other
            limit = 150000
            deduction = min(total, limit)
            
            st.markdown('<div class="calculator-result">', unsafe_allow_html=True)
            st.markdown("#### Calculation Result")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Invested", f"₹{total:,.2f}")
            col2.metric("Deduction Allowed", f"₹{deduction:,.2f}")
            col3.metric("Excess Amount", f"₹{max(0, total - limit):,.2f}")
            
            st.progress(min(total/limit, 1.0))
            st.caption(f"You've used {min(100, (total/limit)*100):.1f}% of Section 80C limit")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif calc_type == "Tax Liability":
        st.markdown("#### Income Tax Liability Calculator")
        
        regime = st.radio("Select Tax Regime:", ["Old Regime", "New Regime"])
        
        gross_income = st.number_input("Gross Total Income (₹)", min_value=0.0, value=1200000.0)
        
        if regime == "Old Regime":
            st.markdown("**Deductions (Chapter VI-A)**")
            col1, col2 = st.columns(2)
            with col1:
                deduction_80c = st.number_input("80C (₹)", min_value=0.0, max_value=150000.0, value=150000.0)
                deduction_80d = st.number_input("80D (₹)", min_value=0.0, max_value=25000.0, value=25000.0)
            with col2:
                deduction_80g = st.number_input("80G (₹)", min_value=0.0, value=0.0)
                other_deductions = st.number_input("Other Deductions (₹)", min_value=0.0, value=0.0)
        
        if st.button("Calculate Tax"):
            # Demo calculation
            standard_deduction = 50000
            
            if regime == "Old Regime":
                total_deductions = deduction_80c + deduction_80d + deduction_80g + other_deductions + standard_deduction
            else:
                total_deductions = standard_deduction
            
            taxable_income = max(0, gross_income - total_deductions)
            
            # Simple slab calculation
            if regime == "Old Regime":
                if taxable_income <= 250000:
                    tax = 0
                elif taxable_income <= 500000:
                    tax = (taxable_income - 250000) * 0.05
                elif taxable_income <= 1000000:
                    tax = 12500 + (taxable_income - 500000) * 0.20
                else:
                    tax = 112500 + (taxable_income - 1000000) * 0.30
            else:
                if taxable_income <= 300000:
                    tax = 0
                elif taxable_income <= 600000:
                    tax = (taxable_income - 300000) * 0.05
                elif taxable_income <= 900000:
                    tax = 15000 + (taxable_income - 600000) * 0.10
                elif taxable_income <= 1200000:
                    tax = 45000 + (taxable_income - 900000) * 0.15
                elif taxable_income <= 1500000:
                    tax = 90000 + (taxable_income - 1200000) * 0.20
                else:
                    tax = 150000 + (taxable_income - 1500000) * 0.30
            
            # Rebate under 87A
            rebate = 0
            if taxable_income <= 500000:
                rebate = min(tax, 12500)
                tax -= rebate
            
            # Cess
            cess = tax * 0.04
            total_tax = tax + cess
            
            st.markdown('<div class="calculator-result">', unsafe_allow_html=True)
            st.markdown(f"#### Tax Liability ({regime})")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Gross Income", f"₹{gross_income:,.2f}")
            col2.metric("Taxable Income", f"₹{taxable_income:,.2f}")
            col3.metric("Total Tax", f"₹{total_tax:,.2f}", delta=f"-₹{rebate:,.2f}" if rebate > 0 else None)
            
            with st.expander("📊 Tax Breakdown"):
                st.write(f"**Income Details:**")
                st.write(f"- Gross Income: ₹{gross_income:,.2f}")
                st.write(f"- Total Deductions: ₹{total_deductions:,.2f}")
                st.write(f"- Taxable Income: ₹{taxable_income:,.2f}")
                st.write(f"\n**Tax Computation:**")
                st.write(f"- Tax as per slabs: ₹{tax + rebate if rebate > 0 else tax:,.2f}")
                if rebate > 0:
                    st.write(f"- Rebate u/s 87A: -₹{rebate:,.2f}")
                st.write(f"- Health & Education Cess (4%): ₹{cess:,.2f}")
                st.write(f"- **Total Tax Liability: ₹{total_tax:,.2f}**")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif calc_type == "Regime Comparison":
        st.markdown("#### Old vs New Regime Comparison")
        
        gross_income = st.number_input("Gross Total Income (₹)", min_value=0.0, value=1200000.0, key="regime_income")
        
        st.markdown("**Old Regime Deductions**")
        col1, col2 = st.columns(2)
        with col1:
            deduction_80c = st.number_input("80C (₹)", min_value=0.0, max_value=150000.0, value=150000.0, key="regime_80c")
            deduction_80d = st.number_input("80D (₹)", min_value=0.0, max_value=25000.0, value=25000.0, key="regime_80d")
        with col2:
            hra_exemption = st.number_input("HRA Exemption (₹)", min_value=0.0, value=100000.0, key="regime_hra")
            other_deductions = st.number_input("Other Deductions (₹)", min_value=0.0, value=0.0, key="regime_other")
        
        if st.button("Compare Regimes"):
            # Calculate for both regimes
            standard_deduction = 50000
            
            # Old regime
            old_deductions = deduction_80c + deduction_80d + hra_exemption + other_deductions + standard_deduction
            old_taxable = max(0, gross_income - old_deductions)
            
            # New regime
            new_deductions = standard_deduction
            new_taxable = max(0, gross_income - new_deductions)
            
            # Calculate tax (simplified)
            # Old regime tax
            if old_taxable <= 250000:
                old_tax = 0
            elif old_taxable <= 500000:
                old_tax = (old_taxable - 250000) * 0.05
            elif old_taxable <= 1000000:
                old_tax = 12500 + (old_taxable - 500000) * 0.20
            else:
                old_tax = 112500 + (old_taxable - 1000000) * 0.30
            
            # New regime tax
            if new_taxable <= 300000:
                new_tax = 0
            elif new_taxable <= 600000:
                new_tax = (new_taxable - 300000) * 0.05
            elif new_taxable <= 900000:
                new_tax = 15000 + (new_taxable - 600000) * 0.10
            elif new_taxable <= 1200000:
                new_tax = 45000 + (new_taxable - 900000) * 0.15
            elif new_taxable <= 1500000:
                new_tax = 90000 + (new_taxable - 1200000) * 0.20
            else:
                new_tax = 150000 + (new_taxable - 1500000) * 0.30
            
            # Add cess
            old_total = old_tax * 1.04
            new_total = new_tax * 1.04
            
            savings = old_total - new_total
            
            st.markdown('<div class="calculator-result">', unsafe_allow_html=True)
            st.markdown("#### Comparison Result")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Old Regime**")
                st.metric("Taxable Income", f"₹{old_taxable:,.2f}")
                st.metric("Total Tax", f"₹{old_total:,.2f}")
            with col2:
                st.markdown("**New Regime**")
                st.metric("Taxable Income", f"₹{new_taxable:,.2f}")
                st.metric("Total Tax", f"₹{new_total:,.2f}")
            with col3:
                st.markdown("**Recommendation**")
                if savings > 0:
                    st.success(f"✅ Choose **New Regime**")
                    st.metric("You Save", f"₹{abs(savings):,.2f}")
                elif savings < 0:
                    st.success(f"✅ Choose **Old Regime**")
                    st.metric("You Save", f"₹{abs(savings):,.2f}")
                else:
                    st.info("Both regimes are equal")
            
            st.markdown('</div>', unsafe_allow_html=True)

def browse_sections_interface():
    """Browse tax sections interface"""
    st.markdown("### 📚 Browse Income Tax Sections")
    
    search_query = st.text_input("Search for a section or topic:", placeholder="E.g., Section 80C or HRA")
    
    if search_query:
        st.info(f"Searching for: {search_query}")
        # Demo results
        st.markdown("#### Search Results")
        
        with st.expander("📄 Section 80C - Deduction in respect of life insurance premia, etc."):
            st.markdown("""
            **Section 80C of Income Tax Act, 1961**
            
            Deduction in respect of life insurance premia, deferred annuity, contributions to provident fund, subscription to certain equity shares or debentures, etc.
            
            **Maximum Deduction:** ₹1,50,000
            
            **Eligible Investments:**
            - Life Insurance Premium
            - Public Provident Fund (PPF)
            - Employee Provident Fund (EPF)
            - Equity Linked Savings Scheme (ELSS)
            - National Savings Certificate (NSC)
            - Tax-saving Fixed Deposits
            - Principal repayment of Home Loan
            - Tuition fees for children (maximum 2 children)
            """)
        
        with st.expander("📄 Section 10(13A) - House Rent Allowance"):
            st.markdown("""
            **Section 10(13A) of Income Tax Act, 1961**
            
            Exemption of House Rent Allowance received by salaried employees.
            
            **Calculation:** Minimum of:
            1. Actual HRA received
            2. 50% of basic salary (metro) or 40% (non-metro)
            3. Rent paid minus 10% of basic salary
            
            **Conditions:**
            - Employee must be paying rent
            - Must not own a house in the city of employment
            - Requires rent receipts if rent > ₹1 lakh per annum
            """)

def export_conversation():
    """Export conversation history"""
    if st.session_state.conversation_history:
        # Create text export
        export_text = "IncomeTaxGPT Conversation Export\n"
        export_text += "=" * 50 + "\n\n"
        
        for msg in st.session_state.conversation_history:
            role = "You" if msg['role'] == 'user' else "IncomeTaxGPT"
            export_text += f"{role} ({msg['timestamp']}):\n"
            export_text += msg['content'] + "\n\n"
            export_text += "-" * 50 + "\n\n"
        
        st.download_button(
            label="📥 Download Conversation",
            data=export_text,
            file_name=f"incometax_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def main():
    """Main application"""
    # Display header
    display_header()
    
    # Display sidebar and get mode
    mode = display_sidebar()
    
    # Initialize system
    if not initialize_system():
        st.error("⚠️ System initialization failed. Please check logs.")
        return
    
    # Route to appropriate interface
    if mode == "💬 Chat Assistant":
        chat_interface()
    elif mode == "🧮 Tax Calculators":
        calculator_interface()
    elif mode == "📚 Browse Sections":
        browse_sections_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> IncomeTaxGPT provides informational guidance only and is not a substitute for professional tax advice. 
    For complex tax situations, please consult a qualified Chartered Accountant or tax professional.</p>
    <p>Built with ❤️ for VIT Capstone Project | Data current as of FY 2023-24</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()