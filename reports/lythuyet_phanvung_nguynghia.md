# Báo cáo: Các Phương pháp Phân vùng Ngữ nghĩa sử dụng Học sâu

## 1. Giới thiệu Phân vùng Ngữ nghĩa (Semantic Segmentation)

Phân vùng ngữ nghĩa là tác vụ gán nhãn cho từng pixel trong ảnh, xác định pixel đó thuộc về lớp đối tượng nào. Khác với phân loại ảnh (chỉ gán nhãn cho toàn bộ ảnh) hay object detection (chỉ xác định bounding box), semantic segmentation yêu cầu hiểu chi tiết về không gian và ngữ nghĩa của từng pixel.

## 2. Các Phương pháp Tiên tiến

### 2.1 U-Net (2015)

**Kiến trúc:**
- Cấu trúc Encoder-Decoder với skip connections
- Encoder: Giảm dần kích thước, tăng depth
- Decoder: Tăng dần kích thước, giảm depth
- Skip connections: Kết nối trực tiếp giữa encoder và decoder layers

**Ưu điểm:**
- Hiệu quả với dataset nhỏ
- Skip connections giúp preserve chi tiết fine-grained
- Kiến trúc đơn giản, dễ hiểu và triển khai
- Phù hợp cho medical imaging

**Nhược điểm:**
- Thiếu context information toàn cục
- Hiệu suất hạn chế trên dataset phức tạp
- Không tận dụng được multi-scale features

### 2.2 DeepLab v3+ (2018)

**Kiến trúc:**
- Atrous Spatial Pyramid Pooling (ASPP)
- Atrous convolution để mở rộng receptive field
- Encoder-decoder structure với Xception backbone

**Ưu điểm:**
- Xử lý tốt objects ở nhiều scales khác nhau
- Atrous convolution giữ được spatial resolution
- ASPP module capture multi-scale context
- State-of-the-art performance trên nhiều benchmarks

**Nhược điểm:**
- Phức tạp trong triển khai
- Yêu cầu nhiều computational resources
- Khó điều chỉnh hyperparameters

### 2.3 PSPNet (Pyramid Scene Parsing Network, 2017)

**Kiến trúc:**
- Pyramid Pooling Module
- Global average pooling ở nhiều scales
- ResNet backbone với dilated convolutions

**Ưu điểm:**
- Tận dụng global context information
- Hiệu quả với scene parsing
- Pyramid pooling giúp capture information ở nhiều levels
- Tốt cho large-scale objects

**Nhược điểm:**
- Chi tiết boundary không sắc nét
- Memory intensive do pyramid pooling
- Hiệu suất kém với small objects

### 2.4 SegNet (2017)

**Kiến trúc:**
- Symmetric encoder-decoder
- Sử dụng pooling indices từ encoder trong decoder
- VGG-16 backbone

**Ưu điểm:**
- Memory efficient
- Giữ được spatial information tốt
- Kiến trúc cân đối và elegant
- Phù hợp cho real-time applications

**Nhược điểm:**
- Hiệu suất lower so với các methods mới hơn
- Thiếu skip connections như U-Net
- Limited expressiveness của VGG backbone

## 3. So sánh các Phương pháp

| Phương pháp | Độ chính xác | Tốc độ | Memory | Complexity | Use Case |
|-------------|--------------|--------|---------|------------|----------|
| U-Net | Trung bình | Nhanh | Thấp | Đơn giản | Medical, Small datasets |
| DeepLab v3+ | Cao | Chậm | Cao | Phức tạp | General purpose, High accuracy |
| PSPNet | Cao | Trung bình | Cao | Trung bình | Scene parsing, Large objects |
| SegNet | Trung bình | Nhanh | Thấp | Đơn giản | Real-time, Resource limited |

## 4. Xu hướng Phát triển

### 4.1 Attention Mechanisms
- Self-attention trong semantic segmentation
- Channel attention và spatial attention
- Non-local networks

### 4.2 Vision Transformers (ViTs)
- SETR (SEgmentation TRansformer)
- Segmenter
- SegFormer

### 4.3 Real-time Methods
- BiSeNet, ICNet
- ENet, ERFNet
- Trade-off giữa accuracy và speed

## 5. Challenges và Hướng nghiên cứu tương lai

### 5.1 Challenges hiện tại:
- Class imbalance problem
- Boundary artifacts
- Multi-scale object handling
- Real-time inference

### 5.2 Hướng nghiên cứu:
- Few-shot và zero-shot segmentation
- Domain adaptation
- 3D semantic segmentation
- Panoptic segmentation (kết hợp semantic và instance)

## 6. Kết luận

Các phương pháp phân vùng ngữ nghĩa đã phát triển từ các kiến trúc đơn giản như U-Net đến các methods phức tạp như DeepLab v3+. Việc lựa chọn phương pháp phù hợp phụ thuộc vào:
- Yêu cầu về độ chính xác
- Constraints về computational resources
- Đặc thù của dataset và application
- Requirements về real-time performance

Trong thực tế, việc kết hợp multiple techniques và fine-tuning cho specific domains thường mang lại kết quả tốt nhất. 