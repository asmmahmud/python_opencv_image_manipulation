import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#
# height, width, rgb = 400, 400, 3
# size_val = width * height * rgb
# a = np.random.randint(100, 200, size=size_val, dtype=np.uint8)
# img = a.reshape(height, width, rgb)
# img[150:155, :] = [200, 0, 0]
# img[:, 275:280] = [50, 50, 200]
# cv2.circle(img, (130, 63), 55, (20, 210, 20), -1)
# cv2.circle(img, (300, 63), 55, (20, 210, 50), -1)
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, 'Naguib al taufique', (0, 130), font, 1, (200, 100, 100), 2, cv2.LINE_AA)
# cv2.imwrite('../resources/img/handcreated_400x400_2.png', img)
# cv2.imshow("Reed", img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img1 = cv2.imread('../resources/img/national-id-card3.jpg')
img2 = cv2.imread('../resources/img/handcreated_400x400_2.png')

# added_img1 = img1 + img2
# added_img2 = cv2.add(img1, img2)
# added_img = np.hstack((img1, np.zeros([400, 5, 3], dtype=np.uint8), img2))
# cv2.imshow("added_img1", added_img1)
# cv2.imshow("added_img2", added_img2)

added_img2 = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
cv2.imshow("added_img2", added_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#
# img = cv2.imread('../resources/img/image-fig1.jpg')
# pic = imageio.imread('../resources/img/image-fig1.jpg')
#
# pic[:, :, 1] = 0
# pic[:, :, 2] = 0
# # cv2.imshow("Blue", pic[:, :, 0])
# # cv2.imshow("Green", pic[:, :, 1])
# cv2.imshow("Reed", pic)
# cv2.waitKey(0)

# plt.title('R channel')
#
# plt.ylabel('Height {}'.format(pic.shape[0]))
#
# plt.xlabel('Width {}'.format(pic.shape[1]))
#
# plt.imshow(pic[:, :, 0])
#
# plt.show()
