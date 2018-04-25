from future import standard_library

standard_library.install_aliases()

import os
import urllib.request
import urllib.error
import urllib.parse
import io
import os.path
import scipy.misc


def geturlimage(url, timeout=10):
    img = None
    ntry = 0
    while ntry < 3:
        try:
            rspns = urllib.request.urlopen(url, timeout=timeout)
            cntnt = rspns.read()
            img = scipy.misc.imread(io.BytesIO(cntnt))
            break
        except urllib.error.URLError as e:
            ntry += 1
            print(type(e))
    return img


path = 'sporco/data'
urllst = {'lena.png':
          'http://sipi.usc.edu/database/misc/4.2.04.tiff',
          'lena.grey.png':
          'http://web.archive.org/web/20070328214632/http://decsai.ugr.es/~javier/denoise/lena.png',
          'barbara.png':
          'http://www.hlevkin.com/TestImages/barbara.bmp',
          'barbara.grey.png':
          'http://web.archive.org/web/20070209141039/http://decsai.ugr.es/~javier/denoise/barbara.png',
          'mandrill.tif':
          'http://sipi.usc.edu/database/misc/4.2.03.tiff',
          'man.grey.tif':
          'http://sipi.usc.edu/database/misc/5.3.01.tiff',
          'kiel.grey.bmp':
          'http://www.hlevkin.com/TestImages/kiel.bmp'}

for key in list(urllst.keys()):
    fnm = os.path.splitext(key)[0]
    dst = os.path.join(path, fnm) + '.png'
    if not os.path.isfile(dst):
        print('Getting %s' % key)
        img = geturlimage(urllst[key])
        if img is not None:
            scipy.misc.imsave(dst, img)
