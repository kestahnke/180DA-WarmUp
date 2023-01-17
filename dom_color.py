import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                    color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv.VideoCapture(0)

# run GUI event loop
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    # create central rectangle
    rect = cv.rectangle(frame,(500,500),(800,200),(0,255,0),3)

    img = cv.cvtColor(rect, cv.COLOR_BGR2RGB)

    img = rect.reshape((rect.shape[0] * rect.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    # build histogram
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    plt.axis("off")
    plt.imshow(bar)
    plt.show()

    fig.canvas.draw()
    fig.canvas.flush_events()

    # show image
    cv.imshow('frame', frame)
    #cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()