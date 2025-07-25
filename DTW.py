import numpy as np


def showmat(s):
    for i in s:
        print(i)
    print()


# 两点之间的距离，为了简便起见，除非是一维数据，否则得出欧氏距离的平方
def distance(a, b):
    if isinstance(a, list) and isinstance(b, list):
        su = 0
        for i, j in zip(a, b):
            su += pow(i - j, 2)
        return su
    return abs(a - b)


# 求斜率，由于求距离时点与点之间的关联较弱，此处取与前后两点连线的斜率的均值
def diff(a, wavenumbers):
    b = []
    b.append((a[1] - a[0]) / (wavenumbers[1] - wavenumbers[0]))
    for c, d, e, f, g, h in zip(a[:-2], a[1:-1], a[2:], wavenumbers[:-2], wavenumbers[1:-1], wavenumbers[2:]):
        b.append((((d - c) / (g - f)) + ((e - d) / (h - g))) / 2)
    b.append((a[-1] - a[-2]) / (wavenumbers[-1] - wavenumbers[-2]))
    return b


# 计算距离矩阵
def dismat(a, b):
    c = []
    for i in a:
        tmp = []
        for j in b:
            tmp.append(distance(i, j))
        c.append(tmp.copy())
    return c


def w_dismat(a, b, width):
    c = []
    d = []
    p = len(a)
    q = len(b)
    if p > q:
        m = q - p
        n = None
        p = -m + width
        r = q - width + 1
        q = width
    else:
        m = None
        n = p - q
        q = -n + width
        r = p - width + 1
        p = width
        if n == 0:
            n = None
    for k, i, j in zip(range(r), a[:r], b[:r]):
        tm = distance(i, j)
        tmpi = [tm]
        tmpj = [tm]
        for x in b[k + 1:k + q]:
            tmpi.append(distance(i, x))
        c.append(tmpi.copy())
        for y in a[k + 1:k + p]:
            tmpj.append(distance(y, j))
        d.append(tmpj.copy())
    for k, i, j in zip(range(width - 1), a[-p + 1:m], b[-q + 1:n]):
        tm = distance(i, j)
        tmpi = [tm]
        tmpj = [tm]
        if q - k != 2:
            for x in b[-q + k + 2:]:
                tmpi.append(distance(i, x))
        c.append(tmpi.copy())
        if p - k != 2:
            for y in a[-p + k + 2:]:
                tmpj.append(distance(y, j))
        d.append(tmpj.copy())
    return c, d


# 计算代价矩阵
def cosmat(a):
    b = []
    c = []
    tmp = []
    tm = []
    tmp.append(a[0][0])
    tm.append(0)
    for i, j in zip(a[0][1:], tmp):
        tmp.append(i + j)
        tm.append(1)
    b.append(tmp.copy())
    c.append(tm.copy())
    for i, j in zip(a[1:], b):
        tmp = []
        tm = []
        tmp.append(i[0] + j[0])
        tm.append(2)
        for p, q, r, s in zip(j[:-1], tmp, j[1:], i[1:]):
            if p <= q and p <= r:
                tmp.append(p + s)
                tm.append(0)
            else:
                if q <= r:
                    tmp.append(q + s)
                    tm.append(1)
                else:
                    tmp.append(r + s)
                    tm.append(2)
        b.append(tmp.copy())
        c.append(tm.copy())
    return b, c


def w_cosmat(a, b):
    c = []
    d = []
    e = []
    f = []
    tmpi = [a[0][0]]
    tmpj = [b[0][0]]
    tmi = [0]
    tmj = [0]
    for i, j in zip(a[0][1:], tmpi):
        tmpi.append(i + j)
        tmi.append(1)
    for i, j in zip(b[0][1:], tmpj):
        tmpj.append(i + j)
        tmj.append(2)
    c.append(tmpi.copy())
    e.append(tmi.copy())
    d.append(tmpj.copy())
    f.append(tmj.copy())
    for i, j, m, n in zip(a[1:], b[1:], c, d):
        if len(m) > 1:
            if len(n) > 1:
                if m[0] <= n[1] and m[0] <= m[1]:
                    tmp = m[0] + i[0]
                    tm = 0
                else:
                    if n[1] <= m[1]:
                        tmp = n[1] + i[0]
                        tm = 1
                    else:
                        tmp = m[1] + i[0]
                        tm = 2
            else:
                if m[0] <= m[1]:
                    tmp = m[0] + i[0]
                    tm = 0
                else:
                    tmp = m[1] + i[0]
                    tm = 2
        else:
            if len(n) > 1:
                if m[0] <= n[1]:
                    tmp = m[0] + i[0]
                    tm = 0
                else:
                    tmp = n[1] + i[0]
                    tm = 1
            else:
                tmp = m[0] + i[0]
                tm = 0
        tmpi = [tmp]
        tmpj = [tmp]
        tmi = [tm]
        tmj = [tm]
        for p, q, r, s in zip(m[1:], tmpi, m[2:], i[1:]):
            if p <= q and p <= r:
                tmpi.append(p + s)
                tmi.append(0)
            else:
                if q <= r:
                    tmpi.append(q + s)
                    tmi.append(1)
                else:
                    tmpi.append(r + s)
                    tmi.append(2)
        if len(tmpi) < len(i):
            if m[-1] <= tmpi[-1]:
                tmpi.append(m[-1] + i[-1])
                tmi.append(0)
            else:
                tmpi.append(tmpi[-1] + i[-1])
                tmi.append(1)
        c.append(tmpi.copy())
        e.append(tmi.copy())
        for p, q, r, s in zip(n[1:], n[2:], tmpj, j[1:]):
            if p <= q and p <= r:
                tmpj.append(p + s)
                tmj.append(0)
            else:
                if q <= r:
                    tmpj.append(q + s)
                    tmj.append(1)
                else:
                    tmpj.append(r + s)
                    tmj.append(2)
        if len(tmpj) < len(j):
            if n[-1] <= tmpj[-1]:
                tmpj.append(n[-1] + j[-1])
                tmj.append(0)
            else:
                tmpj.append(tmpj[-1] + j[-1])
                tmj.append(2)
        d.append(tmpj.copy())
        f.append(tmj.copy())
    return c, d, e, f


# 展开限宽路径矩阵
def pathtran(a, b, x, y):
    ret = np.zeros([y, x], np.int)
    for k, i, j in zip(range(len(a)), a, b):
        for m, n in enumerate(i):
            ret[k][k + m] = n
        for m, n in enumerate(j[1:]):
            ret[k + m + 1][k] = n
    return ret


# 多点映射为1点时寻找中值
def mid(a, wavenumbers):
    if len(a) == 1:
        return a[0]
    m = (wavenumbers[0] + wavenumbers[-1]) / 2
    i = 0
    for j in wavenumbers:
        if j > m:
            break
        i += 1
    return a[i - 1] + ((a[i] - a[i - 1]) * (m - wavenumbers[i - 1]) / (wavenumbers[i] - wavenumbers[i - 1]))


# 映射
def mapping(a, b, wavenumbers):
    x = len(b[0]) - 1
    y = len(b) - 1
    dat = []
    wav = []
    num = 1
    rst = []
    while x != -1 and y != -1:
        if b[y][x] == 2:
            dat.append(a[y])
            wav.append(wavenumbers[y])
            y -= 1
        else:
            if b[y][x] == 1:
                num += 1
                x -= 1
            else:
                dat.append(a[y])
                wav.append(wavenumbers[y])
                da = mid(dat, wav)
                for i in range(num):
                    rst.append(da)
                dat = []
                wav = []
                num = 1
                x -= 1
                y -= 1
    rst.reverse()
    return rst


def DTW(data, wavenumbers, wavenumbers_m, dis=True, dif=True, mean=None, width=0):
    d = np.array(data)
    if mean is None:
        mean = d.mean(0)
        wavenumbers_m = wavenumbers
    else:
        mean = np.array(mean)
    if dis:
        if dif:
            datatu = []
            for i in d:
                j = diff(i, wavenumbers)
                tmp = []
                for p, q in zip(i, j):
                    tmp.append([p, q])
                datatu.append(tmp.copy())
            meantu = []
            j = diff(mean, wavenumbers_m)
            for p, q in zip(mean, j):
                meantu.append([p, q])
        else:
            datatu = d.tolist()
            meantu = mean.tolist()
    else:
        if dif:
            datatu = []
            for i in d:
                j = diff(i, wavenumbers)
                datatu.append(j.copy())
            meantu = diff(mean, wavenumbers_m)
        else:
            return data
    ret = []
    for i, j in zip(datatu, d):
        if width == 0:
            distmat = dismat(i, meantu)
            costmat, path = cosmat(distmat)
        else:
            distmata, distmatb = w_dismat(i, meantu, width)
            costmata, costmatb, patha, pathb = w_cosmat(distmata, distmatb)
            path = pathtran(patha, pathb, len(meantu), len(i))
        # print(i, meantu)
        # showmat(distmat)
        # showmat(costmat)
        # showmat(path)
        ret.append(mapping(j, path, wavenumbers))
    return np.array(ret), mean
