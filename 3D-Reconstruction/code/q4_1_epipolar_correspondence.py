import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', markersize=8, linewidth=2)
        plt.draw()


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE
    # initial for size and others
    h1,w1,c1 = im1.shape
    h2, w2, c2 = im2.shape
    # Experiment with various window sizes.
    window_size = 8
    half_size = window_size//2
    match_distance = 40
    # make sure in range
    window_left = min(half_size,x1)
    window_right = min(half_size,w1-x1-1)
    window_top = min(half_size, y1)
    window_down = min(half_size, h1 - y1 - 1)
    match_dis_top = min(match_distance, y1)
    match_dis_down = min(match_distance, h2 - y1 - 1)

    # (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    v = np.array([x1, y1, 1])
    l = F.dot(v)
    s = np.sqrt(l[0] ** 2 + l[1] ** 2)
    l = l / s
    # it might be beneficial to consider matches for which the distance from (x1, y1) to (x2, y2) is small.
    # must use y to get x! according to line is like vertical line
    y2_line = np.arange(y1-match_dis_top, y1+match_dis_down)
    # from Essential matrix: ax2 + by2 + c = 0
    x2_line = (-(l[1] * y2_line + l[2])/l[0]).astype(int)

    loc_2 = np.vstack((x2_line,y2_line)).T
    # make sure they are in range
    loc_2 = np.array(list(filter(lambda x: half_size <= x[0] < w2-half_size and half_size <= x[1] < h2-half_size, loc_2)))
    # print(loc_2)
    # (2) Search along this line to check nearby pixel intensity
    im1_win = im1[(y1-window_top):(y1+window_down),(x1-window_left):(x1+window_right)]
    # use a Gaussian weighting of the window
    guassian_x = np.linspace(-1, 1, window_left+window_right)
    guassian_y = np.linspace(-1, 1, window_top + window_down)
    X, Y = np.meshgrid(guassian_x,guassian_y)
    dst = np.sqrt(X**2+Y**2)
    gaussian = np.exp(-(dst ** 2 / 2))
    min_dis = float('inf')
    best_x = None
    best_y = None
    for i in range(len(loc_2)):
        x2,y2=loc_2[i]
        # print(x2,y2)
        im2_win=im2[(y2-window_top):(y2+window_down),(x2-window_left):(x2+window_right)]
        # calculating Euclidean distance
        # get diff of two
        diff = im2_win - im1_win
        # use gaussian
        for i in range(c1):
            diff[:, :, i] = diff[:, :, i] * gaussian
        dist = np.linalg.norm(im2_win - im1_win)
        if dist<min_dis:
            min_dis = dist
            best_y = y2
            best_x = x2
        # print(dist)
    x2 = best_x
    y2= best_y
    return x2,y2


if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')
    
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    
    np.savez('q4_1.npz', F, pts1, pts2)
    epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    print(x2,y2)
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)