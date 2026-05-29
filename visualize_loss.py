import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    print("No loss data provided")
    quit()

loss=np.genfromtxt(sys.argv[1], encoding="utf-8")
x=np.arange(len(loss))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss v Epoch")
print(loss)
plt.plot(x,loss)
plt.show()
