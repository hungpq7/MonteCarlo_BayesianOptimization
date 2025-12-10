## Vì sao chọn BoTorch
- Nếu implement from scratch sẽ phải mất công code lại từng thành phần của thuật toán Bayesian Optimization như Gaussian Process, L-BFGS, acqusition functions,... Tốt hơn hết là dùng một thư viện đã có sẵn những bước cơ bản này và thuận tiện cho việc mở rộng.
- BoTorch là thư viện low-level cho phép thực hiện Bayesian Optimization theo từng building block đã chuẩn hóa dựa trên PyTorch. BoTorch được sử dụng rộng rãi trong cộng đồng nghiên cứu và đáp ứng đủ tiêu chí cần tìm.
- Các ưu điểm khác của BoTorch:
    - Tận dụng cơ chế tính đạo hàm tự động của PyTorch, nhờ đó có thể tối ưu việc tìm next best point bằng L-BFGS
    - Hỗ trợ nhiều về Monte Carlo, phù hợp với hướng đi của paper

## Implement BO bằng BoTorch
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

## Acquisition functions
- BoTorch có sẵn rất nhiều acquisition functions, gồm 2 nhóm analytical như EI, PI, UCB và approximation như qEI, qLogEI. Nhóm analytic có dạng toán học cụ thể và tính được bằng các phép tính đại số thông thường. Nhóm approximation cần simulation mới tính được do tính chất phức tạp.
- Paper MLMCBO giới thiệu một kỹ thuật xây acquisition function mới. Việc này thực hiện bằng cách inherit base class trong BoTorch và viết lại hàm `forward` để tính acquisition value.

## Experiment
- So sánh các candidate sau:
    - Analytical EI
    - qEI
    - qLogEI
    - Nested MC Two-step EI
    - MLMC Two-step EI
    - MLMC Three-step EI
- Benchmark:
    - Các hàm benchmark chuyên cho các kỹ thuật optimization, có thể tuỳ chỉnh số chiều, evaluate nhanh và đã biết trước nghiệm tối ưu
    - Thực tế ML model.
- Design:
    - Một lần chạy của mỗi candidate được cấp tối đa 50 lần gọi hàm `f`. Trong đó, 10 lần call được dùng cho initialization.
    - Đối với EI thông thường: mỗi iteration ứng với một function call. Đối với qEI và các biến thể dùng batch, một iteration tối ưu cho q điểm và ứng với q function calls.
    - Mỗi candidate được lặp lại 10 lần chạy, lấy mean và variance để so sánh.
    - Tiêu chí: giá trị tối ưu tìm được và thời gian chạy.
    - Cấu hình: DataBricks instance 4 CPU & 16GB Memmory