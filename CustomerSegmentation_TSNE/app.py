from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
import numpy as np
import os
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns

# Import các module tự tạo
from data_preprocessing import DataPreprocessor
from clustering import CustomerClustering

app = Flask(__name__)

# Đường dẫn đến dữ liệu
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'customer_São_Paulo_2024.csv')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed_customer_data.csv')

# Biến toàn cục để lưu trữ kết quả
preprocessor = None
clustering = None
processed_data = None
cluster_results = None

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess_data():
    """Tiền xử lý dữ liệu"""
    global preprocessor, processed_data
    
    if request.method == 'POST':
        # Khởi tạo bộ tiền xử lý dữ liệu
        preprocessor = DataPreprocessor(DATA_PATH)
        
        # Tiền xử lý dữ liệu
        processed_data = preprocessor.preprocess_data()
        
        # Lưu dữ liệu đã tiền xử lý
        preprocessor.save_processed_data(PROCESSED_DATA_PATH)
        
        return redirect(url_for('preprocessing_results'))
    
    return render_template('preprocess.html')

@app.route('/preprocessing_results')
def preprocessing_results():
    """Hiển thị kết quả tiền xử lý dữ liệu"""
    global preprocessor
    
    if preprocessor is None:
        return redirect(url_for('preprocess_data'))
    
    # Lấy thông tin cơ bản về dữ liệu
    data_info = preprocessor.explore_data()
    
    # Lấy danh sách các biểu đồ EDA
    eda_images = [f for f in os.listdir('static/images/eda') if f.endswith('.png')]
    
    return render_template(
        'preprocessing_results.html',
        data_info=data_info,
        eda_images=eda_images
    )

@app.route('/clustering', methods=['GET', 'POST'])
def perform_clustering():
    """Thực hiện phân cụm"""
    global clustering, cluster_results
    
    if request.method == 'POST':
        # Lấy các tham số từ form
        perplexity = int(request.form.get('perplexity', 30))
        n_clusters = int(request.form.get('n_clusters', 0))
        
        if n_clusters <= 0:
            n_clusters = None
        
        # Khởi tạo đối tượng phân cụm
        clustering = CustomerClustering(data_path=PROCESSED_DATA_PATH)
        
        # Thực hiện phân tích hoàn chỉnh
        cluster_results = clustering.run_complete_analysis(
            perplexity=perplexity,
            n_clusters=n_clusters
        )
        
        return redirect(url_for('clustering_results'))
    
    return render_template('clustering.html')

@app.route('/clustering_results')
def clustering_results():
    """Hiển thị kết quả phân cụm"""
    global clustering, cluster_results
    
    if clustering is None or cluster_results is None:
        return redirect(url_for('perform_clustering'))
    
    # Lấy danh sách các biểu đồ phân cụm
    clustering_images = [f for f in os.listdir('static/images/clustering') if f.endswith('.png')]
    clustering_html = [f for f in os.listdir('static/images/clustering') if f.endswith('.html')]
    
    # Lấy thông tin về các cụm
    cluster_features = cluster_results['cluster_features']
    optimal_clusters = cluster_results['optimal_clusters']
    comparison = cluster_results['comparison']
    
    return render_template(
        'clustering_results.html',
        clustering_images=clustering_images,
        clustering_html=clustering_html,
        cluster_features=cluster_features.to_html(classes='table table-striped'),
        optimal_clusters=optimal_clusters,
        comparison=comparison.to_html(classes='table table-striped')
    )

@app.route('/visualization/<path:filename>')
def visualization(filename):
    """Hiển thị biểu đồ HTML"""
    return send_file(f'static/images/clustering/{filename}')

@app.route('/download_results')
def download_results():
    """Tải xuống kết quả phân cụm"""
    global cluster_results
    
    if cluster_results is None:
        return redirect(url_for('perform_clustering'))
    
    # Lấy dữ liệu đã phân cụm
    data_with_clusters = cluster_results['data_with_clusters']
    
    # Tạo file CSV
    csv_data = data_with_clusters.to_csv(index=False)
    
    # Trả về file CSV
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='customer_segments.csv'
    )

@app.route('/about')
def about():
    """Trang giới thiệu"""
    return render_template('about.html')

@app.route('/compare_methods')
def compare_methods():
    """So sánh t-SNE và PCA"""
    global clustering, cluster_results
    
    if clustering is None or cluster_results is None:
        return redirect(url_for('perform_clustering'))
    
    # Lấy kết quả so sánh
    comparison = cluster_results['comparison']
    
    # Tạo biểu đồ so sánh
    fig = go.Figure()
    
    # Silhouette Score (cao hơn tốt hơn)
    fig.add_trace(go.Bar(
        x=['t-SNE', 'PCA'],
        y=comparison['Silhouette Score'],
        name='Silhouette Score (cao hơn tốt hơn)',
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Davies-Bouldin Index (thấp hơn tốt hơn)
    fig.add_trace(go.Bar(
        x=['t-SNE', 'PCA'],
        y=comparison['Davies-Bouldin Index'],
        name='Davies-Bouldin Index (thấp hơn tốt hơn)',
        marker_color='rgb(26, 118, 255)'
    ))
    
    # Calinski-Harabasz Index (cao hơn tốt hơn)
    fig.add_trace(go.Bar(
        x=['t-SNE', 'PCA'],
        y=comparison['Calinski-Harabasz Index'],
        name='Calinski-Harabasz Index (cao hơn tốt hơn)',
        marker_color='rgb(0, 204, 150)'
    ))
    
    fig.update_layout(
        title='So sánh t-SNE và PCA',
        xaxis_title='Phương pháp',
        yaxis_title='Giá trị',
        barmode='group'
    )
    
    # Chuyển đổi biểu đồ thành JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('compare_methods.html', graphJSON=graphJSON, comparison=comparison.to_html(classes='table table-striped'))

if __name__ == '__main__':
    # Tạo các thư mục cần thiết
    os.makedirs('static/images/eda', exist_ok=True)
    os.makedirs('static/images/clustering', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    app.run(debug=True)
