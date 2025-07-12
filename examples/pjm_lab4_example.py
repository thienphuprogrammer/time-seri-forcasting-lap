"""
DAT301m Lab 4 - Ví dụ sử dụng với dữ liệu PJM thực tế
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pjm_data_analyzer import PJMDataAnalyzer
from src.lab4_interface import DAT301mLab4Interface

def main():
    """
    Ví dụ hoàn chỉnh cho DAT301m Lab 4 với dữ liệu PJM
    """
    print("="*80)
    print("DAT301m Lab 4 - Ví dụ với dữ liệu PJM")
    print("="*80)
    
    # Bước 1: Phân tích dữ liệu có sẵn
    print("\n🔍 BƯỚC 1: PHÂN TÍCH DỮ LIỆU CÓ SẴN")
    print("-" * 50)
    
    analyzer = PJMDataAnalyzer('data')
    available_regions = analyzer.get_available_regions()
    
    print(f"Các khu vực có sẵn: {available_regions}")
    
    # Chọn khu vực để phân tích (ví dụ: PJME)
    if 'PJME' in available_regions:
        region = 'PJME'
    elif 'AEP' in available_regions:
        region = 'AEP'
    elif 'PJM_Load' in available_regions:
        region = 'PJM_Load'
    else:
        region = available_regions[0]
    
    print(f"Sẽ sử dụng khu vực: {region}")
    
    # Bước 2: Phân tích dữ liệu chi tiết
    print(f"\n📊 BƯỚC 2: PHÂN TÍCH DỮ LIỆU CHI TIẾT - {region}")
    print("-" * 50)
    
    # Tải dữ liệu
    data = analyzer.load_region_data(region)
    
    # Lấy thống kê
    stats = analyzer.get_data_statistics()
    print(f"\nThống kê cơ bản:")
    print(f"  - Số bản ghi: {stats['total_records']:,}")
    print(f"  - Khoảng thời gian: {stats['date_range']['duration_years']:.1f} năm")
    print(f"  - Tiêu thụ trung bình: {stats['mw_statistics']['mean']:.0f} MW")
    print(f"  - Tiêu thụ cao nhất: {stats['mw_statistics']['max']:.0f} MW")
    print(f"  - Tiêu thụ thấp nhất: {stats['mw_statistics']['min']:.0f} MW")
    
    # Tạo visualizations
    print(f"\n📈 BƯỚC 3: TẠO BIỂU ĐỒ PHÂN TÍCH")
    print("-" * 50)
    
    plots = analyzer.create_comprehensive_visualizations(
        region=region,
        save_plots=True,
        plot_dir='plots'
    )
    
    # Tạo báo cáo
    report = analyzer.generate_region_report(region, f'reports/{region}_analysis.txt')
    
    # Bước 4: Chuẩn bị cho Machine Learning
    print(f"\n🤖 BƯỚC 4: CHUẨN BỊ CHO MACHINE LEARNING")
    print("-" * 50)
    
    # Tìm file dữ liệu
    data_file = f'data/{region}_hourly.csv'
    if not os.path.exists(data_file):
        print(f"File {data_file} không tồn tại. Tìm kiếm file khác...")
        for file in os.listdir('data'):
            if region in file and file.endswith('.csv'):
                data_file = f'data/{file}'
                break
    
    print(f"Sử dụng file: {data_file}")
    
    # Khởi tạo interface Lab 4
    lab = DAT301mLab4Interface(
        data_path=data_file,
        region=region,
        input_width=24,  # 24 giờ input
        label_width=1,   # dự đoán 1 giờ
        shift=1,         # shift 1 giờ
        random_seed=42
    )
    
    # Chạy workflow hoàn chỉnh
    print(f"\n🚀 BƯỚC 5: CHẠY WORKFLOW HOÀN CHỈNH")
    print("-" * 50)
    
    try:
        # Chạy complete lab (có thể mất thời gian)
        results = lab.run_complete_lab(
            output_dir=f'lab4_results_{region}/',
            save_plots=True,
            multi_step=False,  # Tắt multi-step để nhanh hơn
            create_ensemble=True
        )
        
        print(f"\n✅ HOÀN THÀNH!")
        print(f"Kết quả đã được lưu tại: lab4_results_{region}/")
        
        # Hiển thị kết quả tóm tắt
        print(f"\n📋 TÓM TẮT KẾT QUẢ:")
        print("-" * 30)
        
        for task_name, task_results in results.items():
            if task_name.startswith('task') and 'evaluation_metrics' in task_results:
                print(f"\n{task_name.upper()}:")
                for model_name, metrics in task_results['evaluation_metrics'].items():
                    rmse = metrics.get('RMSE', 'N/A')
                    mae = metrics.get('MAE', 'N/A')
                    print(f"  {model_name:15} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        # Hiển thị câu trả lời
        print(f"\n❓ CÂU TRẢ LỜI LAB:")
        print("-" * 20)
        
        for question, answer in results.get('answers', {}).items():
            print(f"\n{question}:")
            # Hiển thị 200 ký tự đầu
            answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(answer_preview)
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy lab: {e}")
        print("Hãy thử chạy từng task riêng lẻ:")
        
        # Chạy riêng Task 1 (ít lỗi nhất)
        try:
            task1_results = lab.execute_task1(
                datetime_col='Datetime',
                target_col='MW',
                create_plots=True,
                save_plots=True,
                plot_dir='plots'
            )
            print(f"✅ Task 1 hoàn thành: {task1_results}")
        except Exception as e1:
            print(f"❌ Task 1 lỗi: {e1}")

def quick_analysis_example():
    """
    Ví dụ phân tích nhanh
    """
    print("\n" + "="*60)
    print("VÍ DỤ PHÂN TÍCH NHANH")
    print("="*60)
    
    analyzer = PJMDataAnalyzer('data')
    regions = analyzer.get_available_regions()[:3]  # Lấy 3 khu vực đầu
    
    print(f"So sánh {len(regions)} khu vực: {regions}")
    
    # So sánh các khu vực
    comparison = analyzer.compare_regions(
        regions=regions,
        save_plots=True,
        plot_dir='plots'
    )
    
    print("\nKết quả so sánh:")
    print(comparison['statistics'])

def data_exploration_example():
    """
    Ví dụ khám phá dữ liệu
    """
    print("\n" + "="*60)
    print("VÍ DỤ KHÁM PHÁ DỮ LIỆU")
    print("="*60)
    
    analyzer = PJMDataAnalyzer('data')
    
    # Khám phá từng khu vực
    for region in analyzer.get_available_regions()[:2]:  # Chỉ 2 khu vực đầu
        print(f"\n--- Khám phá khu vực {region} ---")
        
        try:
            data = analyzer.load_region_data(region)
            stats = analyzer.get_data_statistics()
            
            print(f"Thời gian: {stats['date_range']['start']} → {stats['date_range']['end']}")
            print(f"Dữ liệu: {stats['total_records']:,} bản ghi ({stats['date_range']['duration_years']:.1f} năm)")
            print(f"Tiêu thụ: {stats['mw_statistics']['mean']:.0f} MW trung bình")
            
            # Tạo báo cáo
            report = analyzer.generate_region_report(region)
            print(f"Báo cáo: {len(report)} ký tự")
            
        except Exception as e:
            print(f"Lỗi xử lý {region}: {e}")

if __name__ == "__main__":
    # Tạo thư mục cần thiết
    os.makedirs('plots', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Chạy ví dụ chính
    main()
    
    # Uncomment để chạy các ví dụ khác
    # quick_analysis_example()
    # data_exploration_example() 