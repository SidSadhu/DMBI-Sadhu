import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocessing import DataPreprocessor
from utils.apriori_analysis import AprioriAnalyzer
from utils.visualization import MarketBasketVisualizer

# Page configuration
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #21808D;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5E5240;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #FCFCF9;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #32B8C6;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🛒 Market Basket Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Using Apriori Algorithm for Association Rule Mining</div>', unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'basket_created' not in st.session_state:
    st.session_state.basket_created = False
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False


def validate_transaction_data(df):
    """
    Validate uploaded transaction data with improved column detection
    
    Returns:
    --------
    tuple: (is_valid, error_message, cleaned_df)
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Display detected columns for debugging
    st.sidebar.write("**Detected Columns:**", list(df.columns))
    
    # Try to find required columns (case-insensitive, flexible matching)
    invoice_col = None
    description_col = None
    quantity_col = None
    country_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Invoice/Transaction ID detection (IMPROVED)
        if any(keyword in col_lower for keyword in ['invoice', 'transaction', 'order', 'receipt', 'bill', 'member']):
            invoice_col = col
        
        # Description/Product/Item detection (IMPROVED)
        elif any(keyword in col_lower for keyword in ['description', 'item', 'product', 'name', 'stockcode', 'sku']):
            description_col = col
        
        # Quantity detection
        elif any(keyword in col_lower for keyword in ['quantity', 'qty', 'amount', 'count']):
            quantity_col = col
        
        # Country/Location detection
        elif any(keyword in col_lower for keyword in ['country', 'region', 'location', 'nation']):
            country_col = col
    
    # If still not found, try alternative approach: use first 2 columns
    if invoice_col is None and len(df.columns) >= 1:
        st.sidebar.warning("⚠️ Invoice column not detected by keywords. Using first column.")
        invoice_col = df.columns[0]
    
    if description_col is None and len(df.columns) >= 2:
        st.sidebar.warning("⚠️ Description column not detected by keywords. Using second or third column.")
        # Try to find the most likely description column
        for col in df.columns[1:]:
            if col != invoice_col and df[col].dtype == 'object':
                description_col = col
                break
        if description_col is None:
            description_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    # Final validation
    if invoice_col is None or description_col is None:
        missing = []
        if invoice_col is None:
            missing.append("InvoiceNo/Transaction ID/Member Number")
        if description_col is None:
            missing.append("Description/Product Name/Item")
        
        error_msg = f"❌ Missing required columns: {', '.join(missing)}\n\n"
        error_msg += f"📋 Detected columns in your file: {list(df.columns)}\n\n"
        error_msg += "Your CSV should have:\n"
        error_msg += "- InvoiceNo (or Transaction ID, Order ID, Member Number, etc.)\n"
        error_msg += "- Description (or Product Name, Item, itemDescription, StockCode, etc.)\n"
        error_msg += "- Quantity (optional)\n"
        error_msg += "- Country (optional)\n\n"
        error_msg += "💡 Tip: Make sure column names are clear (e.g., 'ProductName', 'itemDescription')"
        
        return False, error_msg, None
    
    # Show which columns were mapped
    st.sidebar.success(f"✅ Mapped Columns:")
    st.sidebar.write(f"- Invoice: `{invoice_col}`")
    st.sidebar.write(f"- Description: `{description_col}`")
    if quantity_col:
        st.sidebar.write(f"- Quantity: `{quantity_col}`")
    if country_col:
        st.sidebar.write(f"- Country: `{country_col}`")
    
    # Rename columns to standard names
    rename_map = {
        invoice_col: 'InvoiceNo',
        description_col: 'Description'
    }
    
    if quantity_col:
        rename_map[quantity_col] = 'Quantity'
    if country_col:
        rename_map[country_col] = 'Country'
    
    df_cleaned = df.rename(columns=rename_map)
    
    # Add Quantity column if not present
    if 'Quantity' not in df_cleaned.columns:
        df_cleaned['Quantity'] = 1
    
    # Add Country column if not present
    if 'Country' not in df_cleaned.columns:
        df_cleaned['Country'] = 'Unknown'
    
    # Basic data cleaning
    df_cleaned = df_cleaned.dropna(subset=['InvoiceNo', 'Description'])
    
    # Remove empty strings
    df_cleaned = df_cleaned[df_cleaned['Description'].astype(str).str.strip() != '']
    df_cleaned = df_cleaned[df_cleaned['InvoiceNo'].astype(str).str.strip() != '']
    
    # Convert InvoiceNo to string
    df_cleaned['InvoiceNo'] = df_cleaned['InvoiceNo'].astype(str)
    
    # Check if we have enough data
    if len(df_cleaned) < 10:
        return False, f"❌ Not enough valid transactions after cleaning (found {len(df_cleaned)}, minimum 10 required)", None
    
    st.sidebar.info(f"✅ Loaded {len(df_cleaned)} rows successfully!")
    
    return True, "✅ Data validated successfully!", df_cleaned


def load_sample_data():
    """Generate sample transaction data"""
    np.random.seed(42)
    items = ['bread', 'milk', 'butter', 'eggs', 'cheese', 'yogurt', 'coffee', 'tea', 'sugar', 'flour', 
             'rice', 'pasta', 'tomatoes', 'onions', 'chicken', 'beef', 'fish', 'apples', 'bananas', 'oranges']

    transactions = []
    for i in range(1000):
        invoice = f"INV{i+1:04d}"
        n_items = np.random.randint(2, 6)
        selected_items = np.random.choice(items, size=n_items, replace=False)
        
        for item in selected_items:
            transactions.append({
                'InvoiceNo': invoice,
                'Description': item,
                'Quantity': np.random.randint(1, 5),
                'Country': np.random.choice(['USA', 'UK', 'France', 'Germany'])
            })
    
    return pd.DataFrame(transactions)


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    st.subheader("📁 Data Source")
    
    data_source = st.radio(
        "Choose data source:",
        ["Sample Data", "Upload CSV File"],
        help="Use sample data or upload your own transaction file"
    )
    
    uploaded_file = None
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="File should contain: InvoiceNo/Member_number, Description/itemDescription"
        )
        
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
        else:
            st.info("📊 Upload a CSV file or switch to sample data")
    else:
        st.info("📊 Using built-in sample transaction data")
    
    st.markdown("---")
    
    # Algorithm parameters
    st.subheader("🔧 Algorithm Parameters")
    
    min_support = st.slider(
        "Minimum Support",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Minimum frequency of itemset occurrence"
    )
    st.caption(f"Current value: {min_support:.3f}")
    
    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum probability of consequent given antecedent"
    )
    st.caption(f"Current value: {min_confidence:.2f}")
    
    min_lift = st.slider(
        "Minimum Lift",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Minimum lift value (>1 indicates positive correlation)"
    )
    st.caption(f"Current value: {min_lift:.1f}")
    
    st.markdown("---")
    
    # Run analysis button - FIXED
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Overview", 
    "🔍 Frequent Itemsets", 
    "📈 Association Rules", 
    "💡 Business Insights",
    "📉 Visualizations",
    "📥 Export Results"
])

# Load data based on source
df = None

if data_source == "Upload CSV File" and uploaded_file is not None:
    # Load uploaded file with robust parsing
    try:
        import io
        
        # Read file content
        content = uploaded_file.read()
        
        # Try to decode
        try:
            text_content = content.decode('utf-8-sig')  # Handles BOM
        except:
            text_content = content.decode('latin-1')
        
        # Remove problematic quotes (each line wrapped in quotes)
        lines = text_content.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove leading/trailing quotes and whitespace
            cleaned_line = line.strip().strip('"').strip("'")
            if cleaned_line:  # Skip empty lines
                cleaned_lines.append(cleaned_line)
        
        # Rejoin lines
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Parse as CSV
        df_uploaded = pd.read_csv(io.StringIO(cleaned_content))
        
        # HANDLE DUPLICATE COLUMNS - CRITICAL FIX
        if df_uploaded.columns.duplicated().any():
            st.sidebar.warning("⚠️ Duplicate columns detected. Removing duplicates...")
            # Keep only first occurrence of each column
            df_uploaded = df_uploaded.loc[:, ~df_uploaded.columns.duplicated(keep='first')]
        
        st.sidebar.success(f"✅ File parsed successfully!")
        st.sidebar.write(f"Columns found: {list(df_uploaded.columns)}")
        st.sidebar.write(f"Rows: {len(df_uploaded)}")
        
        # Validate data
        is_valid, message, df_cleaned = validate_transaction_data(df_uploaded)
        
        if is_valid:
            df = df_cleaned
            st.session_state.data_loaded = True
            st.session_state.data_source = "Uploaded File"
            st.session_state.filename = uploaded_file.name
        else:
            st.error(message)
            st.warning("⚠️ Falling back to sample data")
            df = load_sample_data()
            st.session_state.data_loaded = True
            st.session_state.data_source = "Sample Data (Fallback)"
            st.session_state.filename = None
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.sidebar.error(f"Debug info: {type(e).__name__}: {str(e)}")
        st.warning("⚠️ Falling back to sample data")
        df = load_sample_data()
        st.session_state.data_loaded = True
        st.session_state.data_source = "Sample Data (Fallback)"
        st.session_state.filename = None

else:
    # Load sample data
    df = load_sample_data()
    st.session_state.data_loaded = True
    st.session_state.data_source = "Sample Data"
    st.session_state.filename = None

# ADDITIONAL FIX: Remove duplicates from df before storing
if df is not None and df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

# Store in session
st.session_state.df = df

# TAB 1: Data Overview
with tab1:
    # Show data source
    if st.session_state.data_source == "Uploaded File":
        st.success(f"✅ Using uploaded file: **{st.session_state.filename}**")
    elif st.session_state.data_source == "Sample Data (Fallback)":
        st.warning(f"⚠️ Using sample data as fallback due to upload error")
    else:
        st.success(f"✅ Using built-in sample data")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Unique Invoices", f"{df['InvoiceNo'].nunique():,}")
    with col3:
        st.metric("Unique Items", f"{df['Description'].nunique():,}")
    with col4:
        st.metric("Countries", f"{df['Country'].nunique():,}")
    
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("📊 Data Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Column Data Types:**")
        st.dataframe(pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values
        }), use_container_width=True)
    
    with col2:
        st.write("**Missing Values:**")
        missing = df.isnull().sum()
        st.dataframe(pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': (missing.values / len(df) * 100).round(2)
        }), use_container_width=True)
    
    # Data preprocessing
    st.subheader("🧹 Data Preprocessing")
    
    with st.spinner("Cleaning data..."):
        preprocessor = DataPreprocessor(df)
        cleaned_df = preprocessor.clean_data()
        report = preprocessor.get_cleaning_report()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Rows", f"{report['original_rows']:,}")
    with col2:
        st.metric("Cleaned Rows", f"{report['cleaned_rows']:,}")
    with col3:
        st.metric("Rows Removed", f"{report['rows_removed']:,}", 
                 delta=f"-{report['removal_percentage']}%", delta_color="inverse")
    with col4:
        st.metric("Unique Items", f"{report['unique_items']:,}")
    
    # Top items
    st.subheader("🏆 Top 20 Most Frequent Items")
    top_items = preprocessor.get_top_items(20)
    visualizer = MarketBasketVisualizer()
    fig = visualizer.plot_top_items(top_items, n=20)
    st.pyplot(fig)
    
    # Store cleaned data
    st.session_state.cleaned_df = cleaned_df
    st.session_state.preprocessor = preprocessor
    
    # CSV Format Guide
    st.markdown("---")
    st.subheader("📄 CSV File Format Guide")
    
    st.markdown("""
    To upload your own transaction data, your CSV should have these columns:
    
    | Column Name | Required | Description | Example |
    |-------------|----------|-------------|---------|
    | InvoiceNo / Member_number | ✅ Yes | Transaction/Order/Member ID | INV001, ORDER123, 1808 |
    | Description / itemDescription | ✅ Yes | Product/Item name | Bread, Milk, tropical fruit |
    | Quantity | ⚠️ Optional | Number purchased | 1, 2, 3 |
    | Country | ⚠️ Optional | Customer location | USA, UK, France |
    
    **Sample CSV format:**
    ```
    InvoiceNo,Description,Quantity,Country
    INV001,Bread,2,USA
    INV001,Milk,1,USA
    INV002,Bread,1,UK
    INV002,Butter,1,UK
    INV002,Eggs,3,UK
    ```
    
    **Or Groceries format:**
    ```
    Member_number,Date,itemDescription
    1808,21-07-2015,tropical fruit
    1808,21-07-2015,whole milk
    2552,05-01-2015,yogurt
    ```
    
    **Tips:**
    - Use comma-separated values (CSV format)
    - First row should contain column headers
    - InvoiceNo/Member_number groups items in the same transaction
    - Each row represents one item in a transaction
    - Flexible column names (auto-detected)
    """)
    
    # Download sample template
    sample_template = """InvoiceNo,Description,Quantity,Country
INV001,Bread,2,USA
INV001,Milk,1,USA
INV002,Bread,1,UK
INV002,Butter,1,UK
INV003,Eggs,3,France
INV003,Coffee,1,France
INV004,Tea,2,Germany
INV004,Sugar,1,Germany
INV005,Bread,1,USA
INV005,Butter,2,USA"""
    
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_template,
        file_name="transaction_template.csv",
        mime="text/csv",
        help="Download this template and fill with your data"
    )

# TAB 2: Frequent Itemsets - FIXED
with tab2:
    if st.session_state.data_loaded and st.session_state.run_analysis:
        with st.spinner("Creating transaction basket..."):
            preprocessor = st.session_state.preprocessor
            basket = preprocessor.create_basket()
            st.session_state.basket = basket
        
        st.success(f"✅ Basket created with {len(basket)} transactions and {len(basket.columns)} unique items")
        
        with st.spinner("Finding frequent itemsets..."):
            analyzer = AprioriAnalyzer(basket)
            frequent_itemsets = analyzer.find_frequent_itemsets(min_support=min_support)
            st.session_state.analyzer = analyzer
            st.session_state.frequent_itemsets = frequent_itemsets
        
        st.subheader("🔍 Frequent Itemsets")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frequent Itemsets", len(frequent_itemsets))
        with col2:
            st.metric("Average Support", f"{frequent_itemsets['support'].mean():.4f}")
        with col3:
            st.metric("Max Support", f"{frequent_itemsets['support'].max():.4f}")
        
        # Filter by length
        max_length = int(frequent_itemsets['length'].max())
        length_filter = st.multiselect(
            "Filter by Itemset Length",
            options=list(range(1, max_length + 1)),
            default=list(range(1, min(4, max_length + 1)))
        )
        
        if length_filter:
            filtered_itemsets = frequent_itemsets[frequent_itemsets['length'].isin(length_filter)]
        else:
            filtered_itemsets = frequent_itemsets
        
        st.dataframe(
            filtered_itemsets.sort_values('support', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Support distribution
        st.subheader("📊 Support Distribution")
        visualizer = MarketBasketVisualizer()
        fig = visualizer.plot_support_distribution(frequent_itemsets)
        st.pyplot(fig)
        
        # Reset the flag after processing
        st.session_state.run_analysis = False
        
    else:
        st.info("👆 Click '🚀 Run Analysis' in the sidebar to generate frequent itemsets.")

# TAB 3: Association Rules - FIXED  
with tab3:
    # Check if analysis has been run
    if 'analyzer' in st.session_state and 'frequent_itemsets' in st.session_state:
        with st.spinner("Generating association rules..."):
            analyzer = st.session_state.analyzer
            rules = analyzer.generate_rules(metric="confidence", min_threshold=min_confidence)
            
            # Apply lift filter
            rules = rules[rules['lift'] >= min_lift]
            
            st.session_state.rules = rules
    
    # Display results if they exist
    if 'rules' in st.session_state:
        rules = st.session_state.rules
        
        if len(rules) > 0:
            st.success(f"✅ Generated {len(rules)} association rules")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rules", len(rules))
            with col2:
                st.metric("Avg Confidence", f"{rules['confidence'].mean():.4f}")
            with col3:
                st.metric("Avg Lift", f"{rules['lift'].mean():.2f}")
            with col4:
                st.metric("Max Lift", f"{rules['lift'].max():.2f}")
            
            # Display rules
            st.subheader("📋 Association Rules")
            
            # Sort options
            sort_by = st.selectbox("Sort by", ['lift', 'confidence', 'support'], index=0)
            
            display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(
                sort_by, ascending=False
            )
            
            st.dataframe(display_rules, use_container_width=True, height=400)
            
            # Top rules
            st.subheader("🏆 Top 10 Rules by Lift")
            top_10 = rules.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            st.dataframe(top_10, use_container_width=True)
            
            # Interpretation example
            st.subheader("💡 Rule Interpretation Example")
            example_rule = display_rules.iloc[0]
            
            st.markdown(f"""
            **Rule:** `{example_rule['antecedents']}` → `{example_rule['consequents']}`
            
            **Interpretation:**
            - **Support ({example_rule['support']:.4f}):** This combination appears in {example_rule['support']*100:.2f}% of all transactions
            - **Confidence ({example_rule['confidence']:.4f}):** When customers buy `{example_rule['antecedents']}`, there's a {example_rule['confidence']*100:.2f}% chance they also buy `{example_rule['consequents']}`
            - **Lift ({example_rule['lift']:.2f}):** Customers who buy `{example_rule['antecedents']}` are {example_rule['lift']:.2f}x more likely to buy `{example_rule['consequents']}` compared to random chance
            """)
        else:
            st.warning("⚠️ No rules found with current parameters. Try lowering the thresholds.")
    else:
        st.info("👆 Click '🚀 Run Analysis' in the sidebar to generate association rules.")

# TAB 4: Business Insights
with tab4:
    if 'rules' in st.session_state and len(st.session_state.rules) > 0:
        rules = st.session_state.rules
        
        st.header("💡 Business Insights & Recommendations")
        st.markdown("Transform association rules into actionable business strategies")
        
        # Get top rules
        top_rules = rules.nlargest(10, 'lift')
        
        # === SECTION 1: Product Bundling ===
        st.subheader("📦 Product Bundling Strategies")
        st.markdown("**Create product bundles based on frequently purchased together items**")
        
        bundle_col1, bundle_col2 = st.columns([2, 1])
        
        with bundle_col1:
            bundle_recommendations = []
            for idx, rule in top_rules.iterrows():
                if rule['confidence'] >= 0.6 and rule['lift'] >= 1.5:
                    bundle_recommendations.append({
                        'Bundle Name': f"{rule['antecedents']} + {rule['consequents']}",
                        'Confidence': f"{rule['confidence']*100:.1f}%",
                        'Lift': f"{rule['lift']:.2f}x",
                        'Support': f"{rule['support']*100:.1f}%"
                    })
            
            if bundle_recommendations:
                st.dataframe(
                    pd.DataFrame(bundle_recommendations),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No strong bundling opportunities found. Try lowering thresholds.")
        
        with bundle_col2:
            st.markdown("**💰 Pricing Strategy:**")
            st.markdown("""
            - Bundle discount: 10-15%
            - Increases basket size
            - Clears slow-moving stock
            
            **📈 Expected Impact:**
            - +15-25% basket value
            - +10-20% units per transaction
            """)
        
        st.markdown("---")
        
        # === SECTION 2: Cross-Selling Recommendations ===
        st.subheader("🎯 Cross-Selling Recommendations")
        st.markdown("**Suggest complementary products to customers based on purchase history**")
        
        cross_sell_data = []
        for idx, rule in top_rules.head(5).iterrows():
            cross_sell_data.append({
                'When Customer Buys': rule['antecedents'],
                'Recommend': rule['consequents'],
                'Success Rate': f"{rule['confidence']*100:.1f}%",
                'Strength': f"{rule['lift']:.2f}x more likely",
                'Channel': 'Email, Website, App'
            })
        
        st.dataframe(
            pd.DataFrame(cross_sell_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Implementation suggestions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🛒 At Checkout:**")
            st.markdown("""
            - "Customers who bought X also bought Y"
            - Add to cart suggestions
            - One-click upsells
            """)
        
        with col2:
            st.markdown("**📧 Email Marketing:**")
            st.markdown("""
            - Personalized product recommendations
            - "Complete your purchase" campaigns
            - Post-purchase follow-ups
            """)
        
        with col3:
            st.markdown("**📱 Mobile App:**")
            st.markdown("""
            - Push notifications
            - In-app recommendations
            - Personalized home screen
            """)
        
        st.markdown("---")
        
        # === SECTION 3: Store Layout Optimization ===
        st.subheader("🏬 Store Layout Optimization")
        st.markdown("**Optimize physical/virtual store layout based on product associations**")
        
        layout_col1, layout_col2 = st.columns([3, 2])
        
        with layout_col1:
            st.markdown("**📍 Product Placement Strategies:**")
            
            placement_strategies = []
            for idx, rule in top_rules.head(8).iterrows():
                if rule['lift'] >= 1.5:
                    placement_strategies.append({
                        'Product Pair': f"{rule['antecedents']} ↔ {rule['consequents']}",
                        'Placement Strategy': 'Place adjacent or in same aisle',
                        'Expected Lift': f"+{(rule['lift']-1)*100:.0f}% sales"
                    })
            
            st.dataframe(
                pd.DataFrame(placement_strategies),
                use_container_width=True,
                hide_index=True
            )
        
        with layout_col2:
            st.markdown("**🎨 Layout Principles:**")
            st.markdown("""
            **Physical Store:**
            - Adjacent shelf placement
            - End-cap displays
            - Checkout area items
            
            **E-commerce:**
            - Related products section
            - Frequently bought together
            - Smart search results
            """)
        
        st.markdown("---")
        
        # === SECTION 4: Promotional Campaigns ===
        st.subheader("🎁 Promotional Campaign Ideas")
        st.markdown("**Design data-driven promotional campaigns**")
        
        # Generate campaign ideas
        campaign_ideas = []
        for idx, rule in top_rules.head(5).iterrows():
            discount_percent = 10 + (rule['lift'] - 1) * 5
            
            campaign_ideas.append({
                'Campaign Type': f"Buy {rule['antecedents']}, Get discount on {rule['consequents']}",
                'Discount': f"{min(discount_percent, 25):.0f}% off",
                'Target Audience': f"Buyers of {rule['antecedents']}",
                'Expected Conversion': f"{rule['confidence']*100:.1f}%",
                'Priority': '🔥 High' if rule['lift'] >= 2.0 else '⭐ Medium'
            })
        
        st.dataframe(
            pd.DataFrame(campaign_ideas),
            use_container_width=True,
            hide_index=True
        )
        
        # Campaign examples
        st.markdown("**📢 Example Campaign Messaging:**")
        
        if len(top_rules) > 0:
            example_rule = top_rules.iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Email Subject:**  
                "Complete Your Purchase - {example_rule['consequents']} Pairs Perfectly with {example_rule['antecedents']}!"
                
                **Email Body:**  
                "Hi [Customer],
                
                We noticed you recently purchased {example_rule['antecedents']}. 
                
                Did you know {int(example_rule['confidence']*100)}% of our customers also buy {example_rule['consequents']}?
                
                Get 15% off {example_rule['consequents']} this week only!
                
                [Shop Now Button]"
                """)
            
            with col2:
                st.markdown(f"""
                **In-Store Signage:**  
                "Customers who bought {example_rule['antecedents']} also loved {example_rule['consequents']}"
                
                **Mobile Push Notification:**  
                "🎉 Special offer! Add {example_rule['consequents']} to your cart and save 15%"
                
                **Website Banner:**  
                "Trending Bundle: {example_rule['antecedents']} + {example_rule['consequents']} - Save 20%!"
                """)
        
        st.markdown("---")
        
        # === SECTION 5: Business Impact Metrics ===
        st.subheader("📊 Estimated Business Impact")
        st.markdown("**Projected improvements from implementing these recommendations**")
        
        # Calculate metrics
        avg_confidence = rules['confidence'].mean()
        avg_lift = rules['lift'].mean()
        strong_rules = len(rules[rules['lift'] >= 1.5])
        
        impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
        
        with impact_col1:
            st.metric(
                "Cross-Sell Success Rate",
                f"{avg_confidence*100:.1f}%",
                help="Average probability of successful cross-sell"
            )
        
        with impact_col2:
            st.metric(
                "Average Lift",
                f"{avg_lift:.2f}x",
                help="Average increase in likelihood vs random"
            )
        
        with impact_col3:
            st.metric(
                "Strong Associations",
                f"{strong_rules}",
                help="Number of rules with lift > 1.5"
            )
        
        with impact_col4:
            estimated_revenue_increase = min(avg_lift * 10, 30)
            st.metric(
                "Est. Revenue Increase",
                f"+{estimated_revenue_increase:.1f}%",
                help="Projected revenue increase from implementation"
            )
        
        # ROI Calculator
        st.markdown("---")
        st.subheader("💰 ROI Calculator")
        
        roi_col1, roi_col2 = st.columns([1, 1])
        
        with roi_col1:
            st.markdown("**Input Your Business Metrics:**")
            
            monthly_transactions = st.number_input(
                "Monthly Transactions",
                min_value=100,
                max_value=1000000,
                value=10000,
                step=1000,
                help="Average number of transactions per month"
            )
            
            avg_basket_value = st.number_input(
                "Average Basket Value ($)",
                min_value=10.0,
                max_value=10000.0,
                value=50.0,
                step=5.0,
                help="Average transaction value"
            )
            
            implementation_cost = st.number_input(
                "Implementation Cost ($)",
                min_value=0.0,
                max_value=100000.0,
                value=5000.0,
                step=1000.0,
                help="One-time cost to implement recommendations"
            )
        
        with roi_col2:
            st.markdown("**Projected Results:**")
            
            # Calculations
            current_monthly_revenue = monthly_transactions * avg_basket_value
            projected_increase = estimated_revenue_increase / 100
            additional_revenue = current_monthly_revenue * projected_increase
            annual_additional_revenue = additional_revenue * 12
            roi_months = implementation_cost / additional_revenue if additional_revenue > 0 else 999
            
            st.metric(
                "Current Monthly Revenue",
                f"${current_monthly_revenue:,.0f}"
            )
            
            st.metric(
                "Additional Monthly Revenue",
                f"${additional_revenue:,.0f}",
                delta=f"+{projected_increase*100:.1f}%"
            )
            
            st.metric(
                "Annual Additional Revenue",
                f"${annual_additional_revenue:,.0f}"
            )
            
            st.metric(
                "Payback Period",
                f"{roi_months:.1f} months" if roi_months < 24 else "24+ months"
            )
            
            if roi_months <= 6:
                st.success("🎉 Excellent ROI! Implementation highly recommended.")
            elif roi_months <= 12:
                st.info("👍 Good ROI. Implementation recommended.")
            else:
                st.warning("⚠️ Consider optimizing parameters for better ROI.")
        
        # Key Recommendations Summary
        st.markdown("---")
        st.subheader("🎯 Key Action Items")
        
        st.markdown(f"""
        Based on the analysis of your transaction data, here are the **top priority actions**:
        
        1. **Implement Top {min(5, len(top_rules))} Product Bundles**
           - Create bundle offers for highest lift combinations
           - Offer 10-15% discount on bundles
           - Expected impact: +{estimated_revenue_increase:.0f}% revenue
        
        2. **Launch Cross-Sell Email Campaign**
           - Target customers who bought high-frequency items
           - Send personalized recommendations
           - Expected conversion: {avg_confidence*100:.0f}%
        
        3. **Optimize Product Placement**
           - Reorganize store/website layout based on associations
           - Place complementary products together
           - Expected sales lift: {avg_lift:.1f}x
        
        4. **Create Promotional Campaigns**
           - Design "Buy X, Get Y" offers for top rules
           - Run time-limited promotions
           - Track and measure results
        
        5. **Monitor and Iterate**
           - Track implementation results
           - Re-run analysis quarterly
           - Adjust strategies based on performance
        """)
        
    else:
        st.info("👆 Run the analysis first to view business insights.")

# TAB 5: Visualizations
with tab5:
    if 'rules' in st.session_state and len(st.session_state.rules) > 0:
        rules = st.session_state.rules
        visualizer = MarketBasketVisualizer()
        
        st.subheader("📊 Interactive Visualizations")
        
        # Scatter plot
        st.markdown("#### Support vs Confidence (colored by Lift)")
        fig = visualizer.plot_scatter_support_confidence(rules)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top rules bar chart
        st.markdown("#### Top Rules by Lift")
        n_rules = st.slider("Number of rules to display", 5, 30, 10, key="bar_slider")
        fig = visualizer.plot_top_rules_bar(rules, n=n_rules)
        st.plotly_chart(fig, use_container_width=True)
        
        # Network graph
        st.markdown("#### Association Rules Network")
        n_network = st.slider("Number of rules in network", 10, 50, 20, key="network_slider")
        try:
            fig = visualizer.plot_network_graph(rules, top_n=n_network)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create network graph: {e}")
        
        # Heatmap
        st.markdown("#### Metrics Heatmap")
        n_heatmap = st.slider("Number of rules in heatmap", 5, 30, 15, key="heatmap_slider")
        fig = visualizer.plot_heatmap(rules, top_n=n_heatmap)
        st.pyplot(fig)
        
    else:
        st.info("👆 Run the analysis first to view visualizations.")

# TAB 6: Export Results - FIXED
with tab6:
    if 'rules' in st.session_state and len(st.session_state.rules) > 0:
        st.subheader("📥 Export Results")
        
        rules = st.session_state.rules
        frequent_itemsets = st.session_state.frequent_itemsets
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Association Rules")
            
            # Convert to CSV
            csv_rules = rules.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📄 Download Rules (CSV)",
                data=csv_rules,
                file_name="association_rules.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Frequent Itemsets")
            
            # Convert to CSV
            csv_itemsets = frequent_itemsets.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📄 Download Itemsets (CSV)",
                data=csv_itemsets,
                file_name="frequent_itemsets.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Summary report
        st.markdown("#### 📊 Summary Report")
        
        data_info = f"- Data Source: {st.session_state.data_source}"
        if st.session_state.filename:
            data_info += f"\n- Filename: {st.session_state.filename}"
        
        # Get top 10 rules for report
        top_10_rules = rules.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        
        # Create formatted table string
        rules_table = "| # | Antecedents | Consequents | Support | Confidence | Lift |\n"
        rules_table += "|---|-------------|-------------|---------|------------|------|\n"
        for idx, (i, row) in enumerate(top_10_rules.iterrows(), 1):
            rules_table += f"| {idx} | {row['antecedents']} | {row['consequents']} | {row['support']:.4f} | {row['confidence']:.4f} | {row['lift']:.2f} |\n"
        
        summary = f"""
# Market Basket Analysis Report

## Data Information
{data_info}
- Total Transactions: {len(st.session_state.basket)}
- Unique Items: {len(st.session_state.basket.columns)}

## Analysis Parameters
- Minimum Support: {min_support}
- Minimum Confidence: {min_confidence}
- Minimum Lift: {min_lift}

## Results Summary
- Total Frequent Itemsets: {len(frequent_itemsets)}
- Total Association Rules: {len(rules)}
- Average Confidence: {rules['confidence'].mean():.4f}
- Average Lift: {rules['lift'].mean():.2f}

## Top 10 Rules by Lift

{rules_table}

## Business Recommendations

### Product Bundling
Top 5 bundle opportunities based on highest lift values.

### Cross-Selling
Implement personalized recommendations for {len(rules[rules['confidence'] >= 0.5])} strong associations.

### Estimated Impact
- Projected Revenue Increase: +{min((rules['lift'].mean()) * 10, 30):.1f}%
- Strong Associations: {len(rules[rules['lift'] >= 1.5])}
- Average Cross-sell Success: {rules['confidence'].mean()*100:.1f}%
        """
        
        st.download_button(
            label="📄 Download Summary Report (Markdown)",
            data=summary.encode('utf-8'),
            file_name="analysis_report.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### Preview Summary Report")
        st.markdown(summary)
        
    else:
        st.info("👆 Run the analysis first to export results.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #5E5240;'>
    <p>Market Basket Analysis Dashboard | Built with Streamlit | DMBI Project</p>
    <p>Apriori Algorithm Implementation for Association Rule Mining</p>
</div>
""", unsafe_allow_html=True)
