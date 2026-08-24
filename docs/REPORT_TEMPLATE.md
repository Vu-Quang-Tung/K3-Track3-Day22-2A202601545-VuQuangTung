# Preference Alignment Experiment Report (Student Template)

*Instructions: Fill out this report as you complete the lab milestones. Replace all bracketed text `[like this]` with your own findings.*

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: 24
- **Validation issues found**: Dòng 1 bị lỗi cú pháp JSON (unescaped quotes quanh "self-attention").
- **Cleaning steps taken**: Đã escape dấu nháy kép bên trong chuỗi prompt thành \" ở dòng 1.

### Split Strategy
- **Train/Val Ratio**: 0.2 (20% cho tập validation)
- **Leakage Prevention**: Sử dụng `collections.defaultdict` để nhóm các ví dụ theo prompt, đảm bảo mọi bản ghi của cùng một prompt luôn rơi hoàn toàn vào tập train hoặc val. Danh sách các prompt được xáo trộn tất định bằng `random.Random(seed).shuffle`.

## 2. Implementation: DPO & ORPO

### Objective Selection
- **Why this method?**: Đã cài đặt thành công cả DPO và ORPO. DPO yêu cầu reference model để tránh mô hình bóp méo ngôn ngữ gốc, trong khi ORPO sử dụng odds-ratio (log-odds) trực tiếp và phạt lệch mà không cần reference model, tiết kiệm bộ nhớ tính toán.
- **Key Hyperparameters**:
    - `beta`: 0.1 (mặc định thử nghiệm)
    - `lambda_orpo`: 0.1 (mặc định thử nghiệm)

### Numerical Stability
- **Challenges**: Tính log(sigmoid(x)) dễ bị tràn số, hoặc hàm tính _log_odds gọi `np.log1p(-np.exp(logp))` có thể ra -inf nếu logp = 0.
- **Solutions**: Sử dụng `np.logaddexp(0.0, -x)` cho hàm _log_sigmoid. Đối với _log_odds, sử dụng `np.clip(logp, -30.0, -1e-7)` để giới hạn log-prob ở mức an toàn.

## 3. Evaluation Results

### Metrics
| Metric | Value |
|---|---|
| Pairwise Accuracy | 52.08% |
| Final Loss (Mock/Train) | N/A (Chưa tích hợp training loop thực tế) |

### Qualitative Review
- **Prompt**: `[Insert prompt]`
- **Chosen Response**: `[Text]`
- **Rejected Response**: `[Text]`
- **Model Preference**: `[Correct/Incorrect]`

## 4. Discussion & Failure Modes

- **What went well?**: `[observations]`
- **Observed Bias**: `[e.g., Did the model prefer shorter responses regardless of quality?]`
- **Safety**: `[How did the model handle the regression prompts in docs/regression_prompts.md?]`
