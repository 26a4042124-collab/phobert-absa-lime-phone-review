import pandas as pd
import re
import os

def flatten_visfd_data(input_csv_path, output_csv_path):
    """
    Hàm đọc dữ liệu gốc từ UIT-ViSFD và trải phẳng thành định dạng chuẩn cho mô hình ABSA.
    """
    print(f"Đang xử lý file: {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    processed_data = []
    
    for index, row in df.iterrows():
        comment = str(row['comment']).strip()
        label_str = str(row['label']).strip()
        
        # Bỏ qua các dòng không có comment hoặc label
        if pd.isna(row['comment']) or pd.isna(row['label']):
            continue
            
        # Tìm tất cả các cụm nằm trong ngoặc nhọn {}
        # Ví dụ: {BATTERY#Negative} -> BATTERY#Negative
        matches = re.findall(r'\{(.*?)\}', label_str)
        
        for match in matches:
            if '#' in match:
                aspect, sentiment = match.split('#', 1)
                # Đổi tên nhãn tiếng Anh sang tiếng Việt cho thân thiện với UI (Tuỳ chọn)
                sentiment_map = {
                    'Positive': 'Tích cực',
                    'Negative': 'Tiêu cực',
                    'Neutral': 'Trung tính'
                }
                mapped_sentiment = sentiment_map.get(sentiment, sentiment)
                
                processed_data.append({
                    'comment': comment,
                    'aspect': aspect,
                    'sentiment': mapped_sentiment
                })
            else:
                # Xử lý trường hợp {OTHERS} không có #Sentiment
                processed_data.append({
                    'comment': comment,
                    'aspect': match, # Thường là OTHERS
                    'sentiment': 'Trung tính' # Hoặc có thể drop tuỳ chiến lược của bạn
                })
                
    # Tạo DataFrame mới
    processed_df = pd.DataFrame(processed_data)
    
    # Lưu ra thư mục processed
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    processed_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Đã lưu {len(processed_df)} dòng dữ liệu chuẩn vào: {output_csv_path}\n")

if __name__ == "__main__":
    # Đường dẫn (điều chỉnh theo thư mục của bạn)
    # Giả sử file script đang chạy ở thư mục gốc của project
    train_raw = "data/raw/Train.csv"
    dev_raw = "data/raw/Dev.csv"
    test_raw = "data/raw/Test.csv"
    
    train_proc = "data/processed/train_cleaned.csv"
    dev_proc = "data/processed/dev_cleaned.csv"
    test_proc = "data/processed/test_cleaned.csv"
    
    # Chạy hàm xử lý
    flatten_visfd_data(train_raw, train_proc)
    flatten_visfd_data(dev_raw, dev_proc)
    flatten_visfd_data(test_raw, test_proc)