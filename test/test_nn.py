from src.nn import MLP

def test_predictions():
    iterations = 1000
    
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    n(x)

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    
    for _ in range(iterations):    
        # forward pass
        yhat = [n(x) for x in xs]
        loss = sum((yt - yh)**2 for yt, yh in zip(ys, yhat))
        
        # backward pass
        for p in n.parameters():
            p.grad = 0.0
            
        loss.backward()
        
        # update
        for p in n.parameters():
            p.data -= 0.05 * p.grad
  
    tol = 0.02

    for yt, yh in zip(ys, yhat):
        print(yt, yh)
        
        assert abs(yt - yh.data) < tol
  