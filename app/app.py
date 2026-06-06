import streamlit as st
import pandas as pd
import sys
import os
import streamlit.components.v1 as components

# Trỏ đường dẫn để import được các module trong thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import các hàm từ src
from predict import predict_sentiment
from lime_explain import get_lime_explanation

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Demo ABSA & LIME - PhoBERT", page_icon="📱", layout="wide")
st.title("📱 Hệ thống Phân tích Cảm xúc theo Khía cạnh & Giải thích LIME")
st.markdown("Dự án sử dụng **PhoBERT** để phân tích bình luận điện thoại và **LIME** để giải thích mô hình.")

# --- TẠO MENU SIDEBAR ---
menu = st.sidebar.radio("Chọn chức năng:", ["1. Dự đoán Đơn lẻ", "2. Dự đoán Hàng loạt"])

# DANH SÁCH CÁC KHÍA CẠNH (ASPECTS)
ASPECTS = ["Màn hình", "Pin", "Camera", "Hiệu năng", "Thiết kế", "Giá cả", "Chung"]

# ==========================================
# CHỨC NĂNG 1: DỰ ĐOÁN ĐƠN LẺ
# ==========================================
if menu == "1. Dự đoán Đơn lẻ":
    st.header("Nhập bình luận để kiểm tra")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        comment = st.text_area("Nhập bình luận của khách hàng:", height=150, 
                               placeholder="Ví dụ: Màn hình máy này rất đẹp, nhưng pin tụt nhanh quá.")
    with col2:
        aspect = st.selectbox("Chọn Khía cạnh (Aspect) cần phân tích:", ASPECTS)
        analyze_btn = st.button("Phân tích & Giải thích", use_container_width=True, type="primary")
    
    if analyze_btn:
        if not comment.strip():
            st.warning("Vui lòng nhập nội dung bình luận!")
        else:
            with st.spinner('Đang chạy dự đoán qua PhoBERT...'):
                # 1. Gọi hàm dự đoán (Task 29)
                sentiment, confidence = predict_sentiment(comment, aspect)
                
                # Hiển thị kết quả
                st.subheader("Kết quả dự đoán")
                if sentiment == "positive":
                    st.success(f"**{sentiment}** (Độ tin cậy: {confidence:.2f}%)")
                elif sentiment == "negative":
                    st.error(f"**{sentiment}** (Độ tin cậy: {confidence:.2f}%)")
                else:
                    st.info(f"**{sentiment}** (Độ tin cậy: {confidence:.2f}%)")
                
            with st.spinner('Đang tạo biểu đồ giải thích LIME...'):
                # 2. Gọi hàm LIME (Task 30)
                st.subheader(f"Giải thích mô hình bằng LIME cho khía cạnh '{aspect}'")
                lime_html = get_lime_explanation(comment, aspect)
                
                # Render HTML của LIME trực tiếp lên Streamlit
                if lime_html:
                    components.html(lime_html, height=400, scrolling=True)
                else:
                    st.warning("Không thể tạo giải thích LIME lúc này.")

# ==========================================
# CHỨC NĂNG 2: DỰ ĐOÁN HÀNG LOẠT & UPLOAD FILE
# ==========================================
elif menu == "2. Dự đoán Hàng loạt":
    st.header("Tải lên file dữ liệu (CSV hoặc Excel)")
    st.markdown("File cần có ít nhất 2 cột: **Comment** (Nội dung) và **Aspect** (Khía cạnh).")
    
    uploaded_file = st.file_uploader("Kéo thả file vào đây", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Đọc file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Preview dữ liệu:", df.head())
            
            if st.button("Chạy dự đoán hàng loạt", type="primary"):
                with st.spinner('Đang xử lý toàn bộ dữ liệu...'):
                    # Tạo list lưu kết quả
                    predictions = []
                    
                    for index, row in df.iterrows():
                        comment = str(row.get('Comment', ''))
                        aspect = str(row.get('Aspect', 'Chung'))
                        
                        sent, conf = predict_sentiment(comment, aspect)
                        predictions.append(sent)
                        
                    df['Sentiment_Predict'] = predictions
                    
                    st.success("Dự đoán hoàn tất!")
                    st.dataframe(df, use_container_width=True)
                    
                    # Tải file kết quả về
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Tải kết quả xuống (CSV)",
                        data=csv,
                        file_name='ket_qua_absa.csv',
                        mime='text/csv',
                    )
            
            # --- PHẦN LIME CHO DỰ ĐOÁN HÀNG LOẠT ĐÃ ĐƯỢC FIX ---
            st.markdown("---")
            st.markdown("### Xem giải thích LIME cho một dòng cụ thể")
            
            # Giới hạn số nhập vào không được vượt quá số dòng của file
            max_index = len(df) - 1
            row_idx = st.number_input("Nhập số thứ tự dòng (Index) muốn giải thích:", min_value=0, max_value=max_index, value=0, step=1)
            
            if st.button("Giải thích dòng này"):
                with st.spinner("Đang chạy LIME, vui lòng đợi (có thể mất vài giây)..."):
                    try:
                        selected_text = str(df['Comment'].iloc[row_idx])
                        selected_aspect = str(df['Aspect'].iloc[row_idx])
                        
                        st.info(f"**Đang phân tích:**\n- Câu: {selected_text}\n- Khía cạnh: {selected_aspect}")
                        
                        # Gọi đúng hàm get_lime_explanation
                        html_lime = get_lime_explanation(selected_text, selected_aspect)
                        
                        # Hiển thị biểu đồ
                        if html_lime:
                            components.html(html_lime, height=600, scrolling=True)
                        else:
                            st.warning("Không thể tạo giải thích LIME cho dòng này.")
                    except Exception as e:
                        st.error(f"⚠️ Có lỗi xảy ra khi chạy LIME: {e}")
                        
        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {e}")