import torch
import matplotlib.pyplot as plt

# Used to display the results of a training run. Change path as needed

ckpt = torch.load(r"Models/Test7_model.pt", map_location="cpu", weights_only=False)

tl = ckpt["train_loss"]
vl = ckpt["validation_loss"]

print("Training Loss:", tl[-1,1])
print("Validation Loss:", vl[-1,1])

plt.plot(tl[:,0],tl[:,1], label="Training Loss")
plt.plot(vl[:,0],vl[:,1], label="Validation Loss")
plt.title("Loss During Training")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()