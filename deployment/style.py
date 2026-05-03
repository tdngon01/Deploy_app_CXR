import streamlit as st

def load_css():
    st.markdown(
        """
        <style>
        .app-title
        {
            font-size:30px;
            font-weight:bold;
            color:#0f1114;
            margin-bottom:20px;
            text-align:center;
            background:#40ff1e;
            border-radius: 10px;
        }
        .disease-card
        {
            background:#ced2cf;
            border-radius:12px;
            padding:18px 20px;
            margin-bottom:12px;
            border-left:4px solid transparent;
        }
        .disease-card.positive {border-left-color:#e74c3c;}
        .disease-card.negative {border-left-color:#2ecc71;}
        .card-header
        {
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom:12px;
        }
        .card-name
        {
            font-size:15px;
            font-weight:600;
        }
        .badge
        {
            padding:4px 12px;
            border-radius:20px;
            font-size:11px;
            font-weight:700;
            text-transform:uppercase;
        }
        .badge.positive {background:rgba(231,76,60,0.15); color:#e74c3c;}
        .badge.negative {background:rgba(46,204,113,0.15); color:#2ecc71;}
        .progress-bar-container 
        {
            height:8px;
            background:#2d3139;
            border-radius:4px;
            overflow:hidden;
            margin-bottom:8px;
        }
        .progress-bar-fill
        {
            height:100%;
            border-radius:4px;
        }
        .card-footer
        {
            display:flex;
            justify-content:space-between;
            font-size:12px;
            color:#8b949e;
        }
        .card-desc
        {
            font-size:12px;
            color:#6e7681;
            margin-top:10px;
            line-height:1.5;
            padding-top:10px;
            border-top:1px solid #21262d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def prediction_card(class_name, prob, description, is_positive): #thẻ dự đoán
    if is_positive == True:
        status_class = "positive"
    else:
        status_class = "negative"
    status_text = "PHÁT HIỆN"

    if is_positive == True:
        status_text = "PHÁT HIỆN"
    else:
        status_text = "BÌNH THƯỜNG"

    if is_positive == True:
        bar_color = "#e74c3c"
    else:
        bar_color = "#2ecc71"
        
    percent = prob * 100

    st.markdown(
        f"""
        <div class="disease-card {status_class}">
            <div class="card-header">
                <div class="card-name">{class_name}</div>
                <span class="badge {status_class}">{status_text}</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: {percent}%; background: {bar_color};"></div>
            </div>
            <div class="card-footer">
                <span>Xác suất: <strong>{percent:.1f}%</strong></span>
            </div>
            <div class="card-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )