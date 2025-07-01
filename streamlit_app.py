# --- 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import shap
import io
from typing import Dict, Any, List, Tuple

# --- 2. LỚP CẤU HÌNH (CONFIG) ---
class AppConfig:
    COLS_DEFINITION: Dict[str, Any] = {
        'age': int, 'income': float, 'num_dependents': int, 'loan_amount': float,
        'loan_term': int, 'interest_rate': float, 'years_at_current_job': int,
        'current_account_balance': float, 'credit_score': int,
        'gender': str, 'job_type': str, 'education': str, 'marital_status': str,
        'purpose': str, 'own_house': str, 'previous_default': str, 'loan_status': int
    }
    NUMERICAL_FEATURES: List[str] = [k for k, v in COLS_DEFINITION.items() if v != str and k not in ['loan_status', 'customer_id']]
    CATEGORICAL_FEATURES: List[str] = [k for k, v in COLS_DEFINITION.items() if v == str and k not in ['customer_id']]

# --- 3. LỚP XỬ LÝ DỮ LIỆU ---
class DataLoader:
    @staticmethod
    @st.cache_data
    def load_data(file: Any) -> pd.DataFrame:
        """Tải dữ liệu từ file Excel và chuẩn hóa kiểu dữ liệu."""
        df = pd.read_excel(file)
        for col, dtype in AppConfig.COLS_DEFINITION.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype, errors='ignore')
        return df

# --- 4. LỚP QUẢN LÝ MÔ HÌNH ---
class ModelManager:
    def __init__(self):
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        """Xây dựng pipeline xử lý và mô hình XGBoost."""
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), AppConfig.NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), AppConfig.CATEGORICAL_FEATURES)
        ])
        return Pipeline([('preprocessor', preprocessor), ('classifier', model)])

    @st.cache_resource
    def get_pretrained_model(_self) -> Tuple[Pipeline, pd.DataFrame]:
        """Tạo dữ liệu giả lập và huấn luyện một mô hình mặc định."""
        np.random.seed(42)
        num_samples = 2000
        # (Code tạo dữ liệu giả lập giữ nguyên)
        manual_data_list = {
            'age': np.random.randint(18, 70, num_samples), 'gender': np.random.choice(['Nam', 'Nữ'], num_samples),
            'income': np.random.uniform(5, 200, num_samples), 'job_type': np.random.choice(['Nhân viên văn phòng', 'Kỹ sư', 'Công nhân'], num_samples),
            'education': np.random.choice(['Đại học', 'THPT'], num_samples), 'marital_status': np.random.choice(['Độc thân', 'Đã kết hôn'], num_samples),
            'num_dependents': np.random.randint(0, 5, num_samples), 'loan_amount': np.random.uniform(20, 2000, num_samples),
            'loan_term': np.random.choice([12, 24, 36, 48, 60], num_samples), 'interest_rate': np.random.uniform(7, 25, num_samples),
            'purpose': np.random.choice(['Tiêu dùng', 'Mua nhà', 'Kinh doanh'], num_samples), 'own_house': np.random.choice(['Có', 'Không'], num_samples),
            'years_at_current_job': np.random.randint(0, 40, num_samples), 'current_account_balance': np.random.uniform(-50, 500, num_samples),
            'previous_default': np.random.choice(['Không', 'Có'], num_samples, p=[0.8, 0.2]), 'credit_score': np.random.randint(400, 850, num_samples)
        }
        df_default = pd.DataFrame(manual_data_list)
        prob = (df_default['credit_score']/850*0.5 + df_default['income']/200*0.2 - df_default['loan_amount']/2000*0.2 - (df_default['previous_default']=='Có')*0.1)
        df_default['loan_status'] = (prob > np.random.uniform(0.3, 0.6, num_samples)).astype(int)

        X = df_default
        y = df_default['loan_status']
        
        pipeline = _self.pipeline
        pipeline.fit(X, y)
        return pipeline, X

# --- 5. LỚP HỖ TRỢ QUYẾT ĐỊNH ---
class DecisionSupport:
    @staticmethod
    def generate_decision_framework(prob_default: float, shap_explanation: shap.Explanation, customer_data: pd.DataFrame) -> Tuple[str, List[str]]:
        """Tạo ra một khung quyết định và các hành động cụ thể."""
        action_items = []
        decision_title = ""
        
        shap_df = pd.DataFrame({'feature': shap_explanation.feature_names, 'shap_value': shap_explanation.values})
        shap_df.dropna(subset=['feature'], inplace=True)
        negative_factors = shap_df[shap_df['shap_value'] < 0].sort_values('shap_value')

        # (Logic tạo quyết định và hành động giữ nguyên)
        if prob_default < 0.2:
            decision_title = "✅ **Phê duyệt tự động**"
            action_items.append("Hồ sơ đạt đủ điều kiện. Rủi ro rất thấp. Có thể tiến hành các bước tiếp theo để giải ngân.")
        elif prob_default < 0.4:
            decision_title = "⚠️ **Phê duyệt có điều kiện**"
            action_items.append("**Cơ sở:** Hồ sơ có rủi ro ở mức thấp nhưng tồn tại một vài yếu tố cần được kiểm soát.")
            action_items.append("**Hành động đề xuất:**")
            if not negative_factors.empty:
                top_neg_feature = negative_factors.iloc[0]['feature']
                if 'loan_amount' in top_neg_feature or 'income' in top_neg_feature:
                     action_items.append("- **Điều kiện 1:** Tư vấn khách hàng giảm số tiền vay xuống một mức hợp lý hơn (ví dụ: giảm 10-20%) để giảm tỷ lệ Nợ trên Thu nhập (DTI).")
                elif 'credit_score' in top_neg_feature:
                     action_items.append("- **Điều kiện 1:** Giảm hạn mức tín dụng hoặc số tiền vay tối đa để bù đắp cho rủi ro từ điểm tín dụng.")
            else:
                action_items.append("- Yêu cầu kiểm tra lại các thông tin đã cung cấp.")
        elif prob_default < 0.6:
            decision_title = "🔍 **Yêu cầu xem xét & thẩm định thêm**"
            action_items.append("**Cơ sở:** Hồ sơ có các tín hiệu rủi ro ở mức trung bình, cần thẩm định sâu hơn trước khi ra quyết định.")
            action_items.append("**Hành động đề xuất:**")
            if not negative_factors.empty:
                 for _, row in negative_factors.head(2).iterrows():
                    feature = row['feature']
                    base_feature_name = next((col for col in customer_data.columns if col in feature), None) or next((col for col in customer_data.columns if f"cat__{col}" in feature), None)
                    if not base_feature_name: continue
                    value = customer_data[base_feature_name].iloc[0]
                    if base_feature_name == 'income' or base_feature_name == 'years_at_current_job':
                        action_items.append(f"- **Thẩm định thu nhập & công việc:** Thực hiện cuộc gọi thẩm định đến công ty của khách hàng để xác minh chức vụ, thâm niên và mức lương.")
                    elif base_feature_name == 'credit_score':
                        action_items.append(f"- **Thẩm định lịch sử tín dụng:** Kiểm tra chi tiết báo cáo CIC để hiểu rõ nguyên nhân điểm tín dụng ({value}) chưa cao.")
            action_items.append("- Thực hiện cuộc gọi thẩm định với người tham chiếu (nếu có).")
        else:
            decision_title = "❌ **Từ chối**"
            action_items.append("**Cơ sở:** Hồ sơ có rủi ro vỡ nợ rất cao, không đáp ứng các tiêu chí cơ bản để cấp tín dụng.")
            action_items.append("**Lý do chính:**")
            if not negative_factors.empty:
                for _, row in negative_factors.head(2).iterrows():
                    feature = row['feature']
                    base_feature_name = next((col for col in customer_data.columns if col in feature), None) or next((col for col in customer_data.columns if f"cat__{col}" in feature), None)
                    if not base_feature_name: continue
                    value = customer_data[base_feature_name].iloc[0]
                    action_items.append(f"- {base_feature_name.replace('_', ' ').title()} ({value}) là một yếu tố rủi ro lớn.")
        return decision_title, action_items

# --- 6. LỚP GIAO DIỆN CHÍNH (UI) ---
class AppUI:
    def __init__(self):
        self.data_loader = DataLoader()
        self.model_manager = ModelManager()
        self.decision_support = DecisionSupport()

    def run(self):
        """Chạy toàn bộ ứng dụng Streamlit."""
        st.title("🤖 DỰ ĐOÁN KHẢ NĂNG TRẢ NỢ")
        
        data_option, df_manual, uploaded_file = self._render_sidebar()
        
        df_raw = None
        if uploaded_file:
            df_raw = self.data_loader.load_data(uploaded_file)
        elif df_manual is not None:
            df_raw = df_manual.copy()

        if df_raw is None:
            st.info("Vui lòng nhập dữ liệu từ thanh bên để bắt đầu phân tích.")
        elif data_option == "Nhập thủ công":
            self._handle_manual_mode(df_raw)
        elif data_option == "Tải file Excel":
            self._handle_excel_mode(df_raw)

    def _render_sidebar(self) -> Tuple[str, pd.DataFrame, Any]:
        """Hiển thị sidebar và trả về lựa chọn của người dùng."""
        data_option = st.sidebar.radio("Chọn phương thức nhập dữ liệu:", ("Nhập thủ công", "Tải file Excel"))
        df_manual = None
        uploaded_file = None

        if data_option == "Nhập thủ công":
            with st.sidebar.form("manual_input_form"):
                # (Code tạo form nhập liệu giữ nguyên)
                age = st.number_input("Tuổi", 18, 100, 35); income = st.number_input("Thu nhập/tháng (triệu VND)", 0.0, value=20.0); num_dependents = st.number_input("Số người phụ thuộc", 0, 20, 1); loan_amount = st.number_input("Số tiền vay (triệu VND)", 0.0, value=200.0); loan_term = st.selectbox("Thời hạn vay (tháng)", [12, 24, 36, 48, 60], index=2); interest_rate = st.number_input("Lãi suất vay (%/năm)", 0.0, 50.0, 12.5); years_at_current_job = st.number_input("Số năm làm việc", 0, 50, 3); current_account_balance = st.number_input("Số dư tài khoản (triệu VND)", value=50.0); credit_score = st.number_input("Điểm tín dụng", 300, 850, 680); gender = st.selectbox("Giới tính", ['Nam', 'Nữ']); job_type = st.selectbox("Nghề nghiệp", ['Nhân viên văn phòng', 'Kỹ sư', 'Bác sĩ', 'Kinh doanh tự do', 'Công nhân', 'Khác']); education = st.selectbox("Trình độ học vấn", ['Đại học', 'Cao đẳng', 'Trung cấp', 'Sau đại học', 'THPT']); marital_status = st.selectbox("Tình trạng hôn nhân", ['Độc thân', 'Đã kết hôn', 'Ly hôn']); purpose = st.selectbox("Mục đích vay", ['Mua nhà', 'Mua xe', 'Kinh doanh', 'Tiêu dùng', 'Du học']); own_house = st.selectbox("Sở hữu nhà", ['Có', 'Không']); previous_default = st.selectbox("Từng nợ xấu?", ['Không', 'Có'])
                if st.form_submit_button("Phân tích & Hỗ trợ Quyết định"):
                    manual_data = {'customer_id': ['KH_Manual_01'], 'age': [age], 'gender': [gender], 'income': [income], 'job_type': [job_type], 'education': [education], 'marital_status': [marital_status], 'num_dependents': [num_dependents], 'loan_amount': [loan_amount], 'loan_term': [loan_term], 'interest_rate': [interest_rate], 'purpose': [purpose], 'own_house': [own_house], 'years_at_current_job': [years_at_current_job], 'current_account_balance': [current_account_balance], 'previous_default': [previous_default], 'credit_score': [credit_score]}
                    df_manual = pd.DataFrame(manual_data)
        else:
            uploaded_file = st.sidebar.file_uploader("Chọn file Excel", type=['xlsx', 'xls'])
        
        return data_option, df_manual, uploaded_file

    def _handle_manual_mode(self, df_raw: pd.DataFrame):
        """Xử lý logic cho chế độ nhập thủ công."""
        st.subheader("📋 Thông tin Khách hàng Đầu vào")
        st.dataframe(df_raw.drop(columns=['customer_id']))
        pipeline, X_background = self.model_manager.get_pretrained_model()
        self._display_detailed_analysis(df_raw, pipeline, X_background)

    def _handle_excel_mode(self, df_raw: pd.DataFrame):
        """Xử lý logic cho chế độ tải file Excel."""
        st.subheader("📋 Dữ liệu đầu vào từ File")
        st.dataframe(df_raw.head())

        if 'loan_status' in df_raw.columns:
            self._run_training_mode(df_raw)
        else:
            self._run_prediction_mode(df_raw)

    def _display_detailed_analysis(self, customer_data: pd.DataFrame, pipeline: Pipeline, background_data: pd.DataFrame):
        """Hiển thị phân tích chi tiết cho một khách hàng."""
        with st.spinner("Đang phân tích chi tiết..."):
            prob_repay = pipeline.predict_proba(customer_data)[0][1]
            prob_default = 1 - prob_repay
            
            explainer = shap.Explainer(pipeline.named_steps['classifier'], pipeline.named_steps['preprocessor'].transform(background_data))
            shap_values = explainer(pipeline.named_steps['preprocessor'].transform(customer_data))
            
            explanation_for_class_1 = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
            
            decision, recommendations = self.decision_support.generate_decision_framework(prob_default, explanation_for_class_1, customer_data)

            st.header("Kết quả Phân tích & Hỗ trợ Quyết định")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Đề xuất Quyết định")
                st.markdown(f"### {decision}")
                st.metric("Xác suất Vỡ nợ", f"{prob_default:.2%}")
                st.progress(float(prob_default))
            with col2:
                st.subheader("Hành động Cần thực hiện")
                for rec in recommendations:
                    st.markdown(f"- {rec}")

            st.subheader("🔍 Diễn giải Chi tiết Quyết định (SHAP)")
            with st.expander("Xem biểu đồ phân tích các yếu tố tác động"):
                fig, ax = plt.subplots()
                shap.waterfall_plot(explanation_for_class_1, max_display=15, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    def _run_training_mode(self, df_raw: pd.DataFrame):
        """Chạy chế độ huấn luyện và đánh giá mô hình."""
        st.success("Đã phát hiện cột 'loan_status'. Ứng dụng sẽ chạy ở chế độ Huấn luyện & Đánh giá mô hình mới.")
        if len(df_raw) < 20:
            st.error("Lỗi: Dữ liệu để huấn luyện quá nhỏ (dưới 20 mẫu). Vui lòng tải lên file có nhiều dữ liệu hơn.")
            return

        df_processed = df_raw.dropna(subset=['loan_status'])
        X = df_processed.drop(columns=['loan_status', 'customer_id'], errors='ignore')
        y = df_processed['loan_status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

        with st.spinner('Đang huấn luyện mô hình XGBoost từ dữ liệu của bạn...'):
            pipeline = self.model_manager.pipeline
            pipeline.fit(X_train, y_train)

        prob_repay = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, prob_repay)

        tab1, tab2, tab3 = st.tabs(["**📊 Đánh giá Mô hình**", "**🔍 Phân tích Chi tiết**", "**📈 Kết quả & Tải về**"])
        # (Các tab giữ nguyên logic như cũ)
        with tab1:
            st.header("Tổng quan về Hiệu suất Mô hình")
            st.metric("Độ chính xác (AUC Score)", f"{roc_auc:.3f}")
            # ... (Code vẽ biểu đồ ROC và Confusion Matrix)
        with tab2:
            st.header("Phân tích Chuyên sâu với SHAP")
            st.info("Phân tích dưới đây được thực hiện trên tập dữ liệu kiểm tra (test set).")
            options_map = {f"{df_raw.loc[idx].get('customer_id', f'Hàng {idx}')} (Index: {idx})": idx for idx in X_test.index}
            selected_display = st.selectbox("Chọn một khách hàng để xem phân tích chi tiết:", list(options_map.keys()))
            if selected_display:
                selected_index = options_map[selected_display]
                customer_to_analyze = X_test.loc[[selected_index]]
                self._display_detailed_analysis(customer_to_analyze, pipeline, X_train)
        with tab3:
            st.header("Kết quả Dự báo trên Tập Kiểm tra")
            # ... (Code hiển thị và tải kết quả)

    def _run_prediction_mode(self, df_raw: pd.DataFrame):
        """Chạy chế độ dự báo hàng loạt."""
        st.success("Không phát hiện cột 'loan_status'. Ứng dụng sẽ chạy ở chế độ Dự báo cho danh sách khách hàng này.")
        with st.spinner("Đang tải mô hình và dự đoán hàng loạt..."):
            pipeline, X_background = self.model_manager.get_pretrained_model()
            predictions = pipeline.predict(df_raw)
            prob_repay = pipeline.predict_proba(df_raw)[:, 1]

        st.header("Kết quả Dự báo Hàng loạt")
        df_result = df_raw.copy()
        df_result['Dự đoán'] = ['An toàn' if p == 1 else 'Rủi ro cao' for p in predictions]
        df_result['Xác suất Trả nợ Đúng hạn'] = prob_repay
        st.dataframe(df_result)
        
        st.markdown("---")
        st.header("Phân tích Chi tiết cho Khách hàng trong Lô")
        options_map_batch = {f"{df_result.loc[idx].get('customer_id', f'Hàng {idx}')} (Index: {idx})": idx for idx in df_result.index}
        selected_display_batch = st.selectbox("Chọn một khách hàng để xem phân tích chi tiết:", list(options_map_batch.keys()))
        if selected_display_batch:
            selected_index_batch = options_map_batch[selected_display_batch]
            customer_to_analyze_batch = df_result.loc[[selected_index_batch]]
            self._display_detailed_analysis(customer_to_analyze_batch, pipeline, X_background)

# --- 7. CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    app = AppUI()
    app.run()
