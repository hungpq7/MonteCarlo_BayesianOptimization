# 1. Vì sao chọn BoTorch
- **Code lại from scratch?**. Nếu implement from scratch sẽ phải mất công code lại từng thành phần của thuật toán Bayesian Optimization như Gaussian Process, L-BFGS, acqusition functions,... Tốt hơn hết là dùng một thư viện đã có sẵn những bước cơ bản này và thuận tiện cho việc mở rộng.
- **BoTorch and modular design**. BoTorch là thư viện low-level cho phép thực hiện Bayesian Optimization theo từng building block đã chuẩn hóa dựa trên PyTorch. BoTorch được sử dụng rộng rãi trong cộng đồng nghiên cứu và đáp ứng đủ tiêu chí cần tìm.
- Các ưu điểm khác của BoTorch:
    - Tận dụng cơ chế tính đạo hàm tự động của PyTorch, nhờ đó có thể tối ưu việc tìm next best point bằng L-BFGS
    - Hỗ trợ nhiều về Monte Carlo, phù hợp với hướng đi của paper

# 2. Implementation
1. Chuẩn bị trước hàm blackbox `f` cần maximize với số chiều là d; mỗi lần gọi `f` rất tốn kém. Ngoài ra, từng biến đầu vào của `f` cần biết trước `bounds` - khoảng giá trị.
2. Initialize dữ liệu `x_train` `y_train` bằng một số lần gọi f ngẫu nhiên, thường là 10 hoặc 2d lần. Điều này nhằm đảm bảo tính khách quan, giúp quá trình BO không bị quá lệ thuộc vào điểm khởi tạo ban đầu. Bên cạnh random thuần, cũng có thể dùng Sobol để phủ đều không gian đầu vào.
3. Dùng `SingleTaskGP` định nghĩa kiến trúc cho Gaussian Process `model` với bộ tham số $\theta$ mặc định. Learnable parameters của Gaussian Process gồm:
    - Mean function: constant mean $\mu$
    - Kernel function (thường là RBF, Matern): length scale $\ell$, output scale $\sigma_f$
    - Likelihood: noise $\sigma_n$
4. Sử dụng `ExactMarginalLogLikelihood` để lấy likelihood $f(y|\theta)$. Bước này có 2 điểm cần lưu ý:
    - Góc độ toán học: marginal likelihood cho biết khả năng dữ liệu xuất hiện dưới bộ tham số $\theta$. Đây sẽ là hàm mục tiêu để tối ưu trong bước tiếp theo.
    - Góc độ lập trình: trả ra object `mll` gắn liền với object `model` ở bước trên. Object này được tính hoàn toàn bằng PyTorch nên có thể tính được đạo hàm tại điểm bất kỳ bằng `autograd`.
5. Dùng `fit_gpytorch_mll` để tối ưu hàm mục tiêu likelihood bằng thuật toán L-BFGS. Chú ý hàm này nhận input là `mll` nhưng sẽ tác động vào `model`. Nếu không quen với PyTorch thì chỗ này trông có vẻ kỳ cục nhưng thực chất rất native theo ngôn ngữ của PyTorch. Bên trong thực chất gọi `optimizer.step()` sẽ update các tham số của `model` in-place.
6. Định nghĩa acquisition function chọn từ module `botorch.acquisition`. Với GP `model` đã học từ dữ liệu, ta có thể ước lượng acquisition value tại bất cứ điểm x nào.
7. Dùng `optimize_acqf` để tìm nghiệm cho bài toán tối ưu acquisition: tìm next best point (EI) hoặc best q points (với qEI). Append các new points vào `x_train` `y_train` hiện tại.
8. Lặp lại các bước 3-7 cho đến khi hội tụ hoặc hết budget. Chú ý bộ giá trị $\theta$ mới nhất được lưu dưới biến `state_dict`; iteration tiếp theo sẽ bắt đầu từ vị trí này (gọi là warm start) nên tốc độ hội tụ thường nhanh hơn so với bước khởi tạo.

# 3. Acquisition functions
- **Các loại acquisition function**. BoTorch có sẵn rất nhiều acquisition functions, gồm 2 nhóm analytical như EI, PI, UCB và approximation như qEI, qLogEI. Nhóm analytic có dạng toán học cụ thể và tính được bằng các phép tính đại số thông thường. Nhóm approximation cần simulation mới tính được do tính chất phức tạp.
- **Tự tạo acquisition function mới**. Paper MLMCBO giới thiệu kỹ thuật xây acquisition function EI mới. Việc này thực hiện bằng cách inherit base class trong BoTorch và viết lại hàm `forward` để tính acquisition value.

# 4. Experiment setup
### 4.1. Các candidate
- *Random Search*: hoàn toàn không dùng BO, là baseline tốt với tốc độ nhanh
- *Analytical EI*: phiên bản Expected Improvement truyền thống
- *qEI*: thực hiện BO loop theo batch q points, hy sinh độ chính xác đổi lấy thời gian nhanh hơn
- *qLogEI*: biến thể của qEI giảm sai số tính toán nhờ đó cải thiện độ chính xác, nhưng chậm hơn đáng kể
- *Nested MC Two-step EI*: baseline implementation cho các chiến lược lookahead EI, loại khỏi benchmark do chạy quá chậm.
- *MLMC Two-step EI*: phương pháp đuược đề xuất trong paper cho two-step lookahead EI.
- *MLMC Three-step EI*: phương pháp đuược đề xuất trong paper cho two-step lookahead EI.
### 4.2. Benchmark
- Các hàm benchmark chuyên cho các kỹ thuật optimization, evaluate nhanh và đã biết trước nghiệm tối ưu. Reproduce sẽ chọn cấu hình số chiều cho các hàm này khác so với kết quả trong paper.
    - [Branin](https://www.sfu.ca/~ssurjano/branin.html) (d=2)
    - [EggHolder](https://www.sfu.ca/~ssurjano/egg.html) (d=2)
    - [Ackley](https://www.sfu.ca/~ssurjano/ackley.html) (d=4).
    - [Hartmann](https://www.sfu.ca/~ssurjano/hart6.html) (d=6)
- ML algorithms chạy trên dataset thực tế, dùng mẫu 10k bản ghi từ bộ dữ liệu HIGGS. Các thuật toán khác nhau chạy bài toán binary classification với tiêu chí AUROC.
    - DecisionTree: 3 hyperparameters gồm `max_depth`, `min_samples_split` và `min_samples_leaf`
    - ElasticNet: 2 hyperparameters gồm `alpha` và `l1_ratio`
### 4.3. Experiment configuration
- Một lần chạy của mỗi candidate được cấp **tối đa 50 lần gọi hàm** `f`. Trong đó, 10 lần call được dùng cho initialization. Chú ý: Đối với EI thông thường: mỗi iteration ứng với một function call. Đối với qEI và các biến thể dùng batch, một iteration tối ưu cho q điểm và ứng với q function calls.
- Mỗi candidate được lặp lại **10 runs**, lấy median nghiệm tối ưu và thời gian chạy để so sánh.
- Thời gian tối đa cho mỗi run của mỗi candidate là **10 phút**, gấp ~ 200 lần baseline. Nếu vượt quá sẽ bị dừng và không tính kết quả.
- Cấu hình: DataBricks instance 4 CPU & 16GB Memmory

# 5. Experiment results
|  | Random | EI | qEI | qLogEI | 2-step MLMC |
| --- | --- | --- | --- | --- | --- |
| EggHolder (959) | 766 (28.2s) | 818 (43.1s) | 887 (232s) | **888 (165s)** | 840 (344s) |
| Branin (-0.398) | -1.70 (5.8s) | -0.68 (9.1s) | -0.49 (34.8s) | **-0.40 (38.2s)** | -0.63 (152s) |
| Ackley (0.0) | -16.70 (3.3s) | -12.25 (5.0s) | -12.93 (14.2s) | **-2.82 (62.5s)** | -17.49 (17.2s) |
| Hartmann (3.32) | 1.61 (4.3s) | 2.42 (7.0s) | 3.00 (36.7s) | **3.32 (49.7s)** | 2.77 (174s) |
| DecisionTree | 73.73 (14.6s) | 73.87 (16.3s) | 74.05 (55.0s) | **74.07 (266s)** | 73.86 (28.0s) |
| ElasticNet | 69.030 (12.2s) | 69.032 (12.9s) | **69.037 (75.3s)** | 69.037 (169s) | 69.007 (743s) |

### 5.1. Nhật ký thực nghiệm
- Theo kế hoạch, các thí nghiệm sẽ được cho **chạy song song** để tiết kiệm thời gian. Nhóm giả định rằng thời gian trung bình khi chạy song song sẽ chậm hơn so với chạy tuần tự nên việc so sánh vẫn giữ tính khách quan.
- Tuy nhiên khi thử nghiệm với các phương pháp cần tính toán nặng, dùng Monte-Carlo nhiều; nếu chạy song song sẽ chậm hơn đáng kể. Do vậy setup thí nghiệm sẽ quay về **chạy tuần tự**.
- Ban đầu nhóm không đựa **Random Search** vào danh sách so sánh vì cho rằng nó quá đơn giản. Tuy nhiên sau khi chạy thử nghiệm, nhóm nhận thấy Random Search có kết quả chấp nhận được trong thời gian cực ngắn. Do vậy Random Search được thêm vào bảng kết quả để làm baseline.
- Có 2 ứng viên không chạy thành công trên một số bài toán benchmark do tốn quá nhiều thời gian: Nested MC Two-step EI và MLMC Three-step EI. Nguyên nhân chính là do việc tính toán Monte-Carlo quá nặng nề. Do vậy hai ứng viên này bị loại khỏi bảng kết quả cuối cùng.

### 5.2. Kết luận
- qLogEI là phương pháp tốt nhất, thể hiện ở việc một số bài toán thậm chí đã tiệm cận global maximum hay bỏ xa các ứng viên. Thời gian chạy tuy chậm hơn EI hay qEI nhưng hoàn toàn đủ nhanh.
- qEI chính xác hơn EI và chậm hơn; kết quả này có vẻ ngược với lý thuyết.
- Nếu bạn muốn phương pháp nhanh với độ chính xác vừa đủ, hãy chọn EI. Nếu muốn độ chính xác cao với thời gian chấp nhận được, hãy chọn qLogEI.
- MLMC Two-step EI có độ chính xác khá cạnh tranh, tuy nhiên thời gian chạy khá lâu. 
- Paper đã cung cấp một hướng đi mới mẻ trong việc xây dựng acquisition function cho Bayesian Optimization. Tuy nhiên để áp dụng rộng rãi trong thực tế, cần có thêm các nghiên cứu để tối ưu việc tính toán Monte-Carlo nhằm giảm thời gian chạy.
- Phạm vi của paper gói gọn trong EI acquisition function. Lĩnh vực derivative-free optimization còn rất nhiều khía cạnh như:
    - Các acquisition function khác (PI, UCB, etc)
    - Các cách tiếp cận non-BO khác (TPE, Bandit, Evolutionary, etc)