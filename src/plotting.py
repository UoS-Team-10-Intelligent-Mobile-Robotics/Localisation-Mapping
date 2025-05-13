from plot_feeg6043 import show_information
import numpy as np

info_vec = np.loadtxt("infovec.txt", delimiter=",")
info_vec = info_vec.reshape(info_vec.shape[0], 1)
info_mac = np.loadtxt("infomac.txt", delimiter=",")
print(info_vec)
print(info_mac)

show_information(info_vec, 185, 3, 4, 2, display_type="intensity")
show_information(info_mac, 185, 3, 4, 2, display_type="intensity")
show_information(info_vec, 185, 3, 4, 2, display_type="source")
show_information(info_mac, 185, 3, 4, 2, display_type="source")

print(np.allclose(info_mac, info_mac.T))
print(np.linalg.eigvalsh(info_mac) > 0)  # All must be True
eigvals = np.linalg.eigvalsh(info_mac)
print(eigvals)
