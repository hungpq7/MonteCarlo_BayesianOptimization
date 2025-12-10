## Vì sao chọn BoTorch
- Nếu implement from scratch sẽ phải mất công code lại từng thành phần của thuật toán Bayesian Optimization như Gaussian Process, L-BFGS, acqusition functions,... Tốt hơn hết là dùng một thư viện đã có sẵn những bước cơ bản này và thuận tiện cho việc mở rộng.
- BoTorch là thư viện low-level cho phép thực hiện Bayesian Optimization theo từng building block đã chuẩn hóa dựa trên PyTorch. BoTorch được sử dụng rộng rãi trong công động nghiên cứu và đáp ứng đủ tiêu chí cần tìm.
- Các ưu điểm khác của BoTorch:
    - Tận dụng cơ chế tính đạo hàm tự động của PyTorch, nhờ đó có thể tối ưu việc tìm next best point bằng L-BFGS
    - Hỗ trợ nhiều về Monte Carlo, phù hợp với hướng đi của paper

## Cách implement BO bằng BoTorch
- Chuẩn bị trước hàm blackbox `f` cần maximize với số chiều là d. Giả sử mỗi lần gọi `f` rất tốn kém. Ngoài ra, từng biến đầu vào của `f` cần biết trước `bounds` - khoảng giá trị.
- Initialize BO bằng một số lần gọi f ngẫu nhiên, thường là 10 lần. Điều này nhằm đảm bảo tính khách quan, giúp quá trình BO không bị quá lệ thuộc vào điểm khởi tạo ban đầu.
- Train GaussianProcess `model` từ các điểm khởi tạo bằng `SingleTaskGP`. Model này ban đầu sẽ xây dựng prior theo một vài tham số thống kê, sau đó học posterior từ các điểm khởi tạo.
- Sử dụng `ExactMarginalLogLikelihood` để lấy marginal likelihood. Object `mll` thu được sẽ gắn liền với object `model` ở bước trên. Cập nhật tham số cho GP bằng hàm `fit_gpytorch_mll`.
- Model GP hiện tại cho phép tính acqusition value tại bất kỳ điểm x nào. Ngoài ra cũng biết đạo hàm (nhờ autograd) hoặc estimate (do hàm surrogate rẻ), từ đó giải bài toán tìm điểm tối ưu tiếp theo bằng L-BFGS.
- Append điểm tối ưu tiếp theo vào tập train và tiếp tục cập nhật posterior cho surrogate model. Chú ý các lần cập nhật này được tiếp tục từ giá trị tham số trước đó nên sẽ nhanh hơn so với khi khởi tạo.
- Lặp lại cho đến khi hết budget hoặc hội tụ.

## Acquisition functions
- BoTorch có sẵn rất nhiều acquisition functions, gồm 2 nhóm Analytical như EI, PI, UCB và Monte-Carlo như qEI, qLogEI. Nhóm Analytical có dạng toán học cụ thể và tính được bằng các phép tính đại số thông thường. Nhóm Monte-Carlo cần simulation mới tính được do tính chất phức tạp.
- Paper nhóm đang tìm hiểu giới thiệu một kỹ thuật xây acquisition function mới. Việc này thực hiện bằng cách inherit base class trong BoTorch và viết lại hàm foward để tính acquisition value.

## Experiment
- So sánh các candidate sau:
    - Analytical EI
    - qEI
    - qLogEI
    - Nested MC Two-step EI
    - MLMC Two-step EI
    - MLMC Three-step EI
- Benchmark:
    - Các hàm benchmark chuyên cho các kỹ thuật optimization, có thể tuỳ tỉnh số chiều, evaluate nhanh và đã biết trước nghiệm tối ưu
    - Thực tế ML model.
- Design:
    - Một lần chạy của mỗi candidate được cấp tối đa 50 lần gọi hàm `f`. Trong đó, 10 lần call được dùng cho initialization.
    - Đối với EI thông thường: mỗi iteration ứng với một function call. Đối với qEI và các biến thể dùng batch, một iteration tối ưu cho q điểm và ứng với q function calls.
    - Mỗi candidate được lặp lại 10 lần chạy, lấy mean và variance để so sánh.
    - Tiêu chí: giá trị tối ưu tìm được và thời gian chạy.
    - Cấu hình: