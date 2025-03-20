#1) model design (inpuyt, outputs size,forward pass)
#2) construct loss and optimizer
#3)Training loop
# -forward pass: comput eprediction
# -backward pass: gradients
# -update the weights

import torch
import torch.nn as nn #nueral network for loss 
#f= w*x
#f=2*x
 
X=torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test=torch.tensor([5],dtype=torch.float32)



n_samples, n_features=X.shape
print(n_samples, n_features)

input_size = n_features
output_size=n_features

model = nn.Linear(input_size,output_size)


print(f'prdiction before traininig : f(10) =  {model(X_test).item():.3f}')

#training
learning_rate=0.01
n_iters=100

loss=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)


for epoch in range (n_iters):
    #prediction forward pass
    y_pred= model(X)

    
    #loss
    l= loss(Y,y_pred)

    
    #backward pass
    l.backward()
    
    optimizer.step()
  

    #zewro thr gradient
    optimizer.zero_grad()
        
    
    if epoch % 10 ==0:
        [w,b]=model.parameters()
        print(f"epoch {epoch+1}: weight :{w[0][0].item():.3f}, loss: {l:.8f}")
        
print (f'pediction after trainin: f(10)= {model(X_test).item():.3f}')