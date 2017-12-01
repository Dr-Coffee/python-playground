import numpy as np
import cv2
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

len_shelf_bound = 1500
space_between_products = 10
min_product_size = 100
max_product_size = 300
mean_number_products = 3
common_cov = 0.3
img_height = 1000
product_height = 300
font = cv2.FONT_HERSHEY_SIMPLEX
max_space_between_products = space_between_products * 3

def constructPlanogram():
    len = 0
    id_product = 0
    planogram = []
    while len < len_shelf_bound:
        n_product = int(np.random.normal(mean_number_products, common_cov, 1)[0])
        len_product = int(np.random.uniform(min_product_size, max_product_size, 1)[0])
        planogram.append([id_product, len_product, n_product])#[id, len, num]
        len = len + n_product * len_product + space_between_products
        id_product += 1
    return planogram

def constructRealityWithOneKindOfProductsMissing(planogram, shift, cov_noise):
    n_product = len(planogram)
    id_missing = int(np.random.uniform(1, n_product, 1)[0])
    #id_missing = 0
    realogram = [] #[id, begin, end]
    begin = shift
    for i in range(n_product):
        for j in range(planogram[i][2]):
            if i == id_missing:
                realogram.append([i, begin, begin + planogram[i][1], 0])
            else:
                begin = begin + abs(int(np.random.normal(0, cov_noise, 1)[0]))
                realogram.append([i,begin,begin+planogram[i][1], 1])
            begin = begin + planogram[i][1]
        begin = begin + space_between_products
    return realogram, id_missing

def constructReality(planogram, shift, cov_noise, missing_rate = 0.8):
    n_product = len(planogram)
    realogram = []
    begin = shift
    for i in range(n_product):
        for j in range(planogram[i][2]):
            if bernoulli.rvs(missing_rate, size=1)[0] == 1:
                begin = begin + abs(int(np.random.normal(0, cov_noise, 1)[0]))
                realogram.append([i, begin, begin + planogram[i][1], 1])
            else: #Missing
                realogram.append([i, begin, begin + planogram[i][1], 0])
            begin = begin + planogram[i][1]
        begin = begin + space_between_products
    return realogram

def findShelfLen(planogram):
    shelf_len = 0
    for i in range(len(planogram)):
        shelf_len = shelf_len + planogram[i][1] * planogram[i][2]
    return shelf_len

def visualization(planogram, realogram, gaps):
    shelf_len = findShelfLen(planogram)
    padding = max_product_size * 2
    img = np.zeros((img_height, shelf_len + padding, 3), np.uint8)
    for i in range(len(gaps)):
        cv2.rectangle(img, (gaps[i][0] + int(padding / 2), int(img_height - 99) - product_height),
                      (gaps[i][1] + int(padding / 2), int(img_height - 99)), (100, 100, 100), -1)
    begin = 0
    for i in range(len(planogram)):
        for j in range(planogram[i][2]):
            cv2.rectangle(img, (begin + int(padding/2), int(img_height / 2 - 99) - product_height),
                          (begin + int(padding/2) + planogram[i][1], int(img_height / 2 - 99)), (0, 255, 0), 3)
            cv2.putText(img, str(planogram[i][0]), (begin + int(padding/2), int(img_height / 2 - 300)),
                        font, 4, (255, 255, 255), 2, cv2.LINE_AA)
            begin = begin + planogram[i][1]
        begin = begin + space_between_products
    for i in range(len(realogram)):
        if(realogram[i][3]>0):
            cv2.rectangle(img, (realogram[i][1] + int(padding / 2), int(img_height - 99) - product_height),
                          (realogram[i][2] + int(padding / 2), int(img_height - 99)), (0, 255, 0), 3)
            cv2.putText(img, str(realogram[i][0]), (realogram[i][1] + int(padding / 2), int(img_height - 300)),
                        font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        #else:
        #    cv2.rectangle(img, (begin + int(padding/2), int(img_height / 2 - 99) - product_height),
        #                  (begin + int(padding/2) + planogram[i][1], int(img_height / 2 - 99)), (0, 255, 0), 3)
        #    cv2.putText(img, str(planogram[i][0]), (begin + int(padding/2), int(img_height / 2 - 300)),
        #                font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (int(padding / 2), int(img_height / 2 - 100)),
             (int(shelf_len + padding / 2)+50, int(img_height / 2 - 100)), (0, 0, 255), 3)
    cv2.line(img, (int(padding / 2), int(img_height - 100)),
             (int(shelf_len + padding / 2)+50, int(img_height - 100)), (0, 0, 255), 3)
    #
    img = cv2.resize(img, (int(0.5 * (len_shelf_bound + padding)), int(0.5 * img_height)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("1", img)
    cv2.waitKey(0)
    #return img

def findGap(realogram, shelf_len):
    products = []
    for i in range(len(realogram)):
        if realogram[i][3] > 0:
            products.append(realogram[i])
    gap = []
    if products[0][1] - 0 > max_space_between_products:
        gap.append([0,products[0][1]])
    for i in range(len(products)-1):
        if products[i+1][1] - products[i][2] > max_space_between_products:
            gap.append([products[i][2], products[i+1][1]])
    if shelf_len - products[len(products)-1][2] > max_space_between_products:
        gap.append([products[len(products)-1][2], shelf_len])
    return gap

def overlap(begin, end, pool):
    num = 0
    for i in range(begin, end):
        if pool[i] == True:
            num = num + 1
    o = num / (end-begin)
    #print(o)
    return o

def missingDetection(planogram, gap, th):
    pool = np.zeros(findShelfLen(planogram)+10000, np.bool)
    for g in gap:
        for i in range(g[0], g[1]):
            pool[i] = True
    missing_product = []
    begin = 0
    for i in range(len(planogram)):
        for j in range(planogram[i][2]):
            if overlap(begin, begin + planogram[i][1], pool) > th:
                missing_product.append([planogram[i][0], begin, begin + planogram[i][1]])
            begin = begin + planogram[i][1]
        begin = begin + space_between_products
    return missing_product

def performanceEva(planogram, realogram, missing):
    gt = np.zeros(len(planogram))
    re = np.zeros(len(planogram))

    for p in realogram:
        if p[3] == 0:
            gt[p[0]] = gt[p[0]] + 1
    for p in missing:
        re[p[0]] = re[p[0]] + 1

    tp = 0
    fp = 0
    fn = 0
    r = 0
    w = 0
    for i in range(len(planogram)):
        if gt[i] > 0 and re[i] > 0:
            if gt[i] == re[i]:
                tp = tp + gt[i]
                r = r + 1
            if gt[i] > re[i]:
                fn = fn + gt[i] - re[i]
                w = w + 1
            if gt[i] < re[i]:
                fp = fp + re[i] - gt[i]
                w = w + 1

    if r+w == 0:
        return 1
    else:
        return r/(r+w)
    #return tp/(tp+fp+0.0001), tp/(tp+fn+0.0001)

def analysisProductsMoving():
    times = 100
    #pr = np.zeros(200)
    #rc = np.zeros(200)
    r = np.zeros(200)
    for noise_cov in range(10,200):
        #print("----")
        print(noise_cov)
        for i in range(times):
            planogram = constructPlanogram()
            realogram = constructReality(planogram, 0, noise_cov)
            gaps = findGap(realogram, findShelfLen(planogram))
            missing = missingDetection(planogram, gaps, 0.5)
            temp = performanceEva(planogram, realogram, missing)
            r[noise_cov] = r[noise_cov] + temp
            #pr[noise_cov] = pr[noise_cov] + temp_pr
            #rc[noise_cov] = rc[noise_cov] + temp_rc
        #pr[noise_cov] = pr[noise_cov]/times
        #rc[noise_cov] = rc[noise_cov]/times
        r[noise_cov] = r[noise_cov]/times
        #print(pr)
        #print(rc)
    return r

r = analysisProductsMoving()
plt.plot(range(200), r)
plt.show()

exit(0)
planogram = constructPlanogram()
#realogram = constructRealityWithOneKindOfProductsMissing(planogram, 0, 10)
print(planogram)
realogram = constructReality(planogram, 20, 10)
print(realogram)
gaps = findGap(realogram, findShelfLen(planogram))
print(gaps)
missing = missingDetection(planogram, gaps, 0.5)
print(missing)
print(performanceEva(planogram, realogram, missing))

visualization(planogram, realogram, gaps)


