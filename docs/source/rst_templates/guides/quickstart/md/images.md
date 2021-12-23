
## Work with images from API and host 

Read image from host:

    import supervisely_lib as sly
    image = sly.image.read(os.path.join(os.getcwd(), 'new_image.jpeg'))
    print(type(image))
    print(image.shape)

    <class 'numpy.ndarray'>
    (800, 1067, 3)

Get information about image from API by it ID:

    address = 'https://app.supervise.ly'
    token = 'P78DuO37grwKNbGikDso72gphdCICDsiTXflvSGVEiendUhnJz93Pm48KKPAlgh2k68TPIAR7LPW1etGPiATM1ZOQL8iFVfWjt8gUphxps3IOSicrm6m0gv2cQh3lfww'
    api = sly.Api(address, token)
    image_info = api.image.get_info_by_id(93907659)
    print(image_info)

    ImageInfo(id=93907659, name='IMG_6896.jpeg', link=None, hash='n3+8xGisUeHa6BF+ndaHdfH7Mm9XJaUTFbtPIZ5te7Y=', mime='image/jpeg', ext='jpeg', size=192651, width=1067, height=800, labels_count=9, dataset_id=143202, created_at='2019-08-02T13:11:35.493Z', updated_at='2019-08-02T13:20:31.926Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/r/O/kn/g101i1Dxb7tMoB804wudemjxMraoAjoQ8OpjtCItJUWqySTeItDERtiz0REAnW9IETqIE4gIgOiTM91UnK41dWIUubgwCM6b8JFnuIIU6MeEgFmabMrWxylfKkif.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/r/O/kn/g101i1Dxb7tMoB804wudemjxMraoAjoQ8OpjtCItJUWqySTeItDERtiz0REAnW9IETqIE4gIgOiTM91UnK41dWIUubgwCM6b8JFnuIIU6MeEgFmabMrWxylfKkif.jpg')

If connection to the site is not established or image with given ID not exist raise RetryError.

Get information about images from API by its IDs:

    images_info = api.image.get_info_by_id_batch([193526437, 176024256])
    for image_info in images_info:
        print(image_info)

    ImageInfo(id=193526437, name='image_name.jpg', link=None, hash='ilKYs4pyP47YlihgiZIuVZU7/3MqaI4rJsCgfPpQqaM=', mime='image/jpeg', ext='jpeg', size=193755, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-09T14:32:08.712Z', updated_at='2021-02-09T14:32:08.712Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/B/5/a4/1koN6ouNtKzS2OTaEkZQ0lZuTxo91j7N3i2BhJVAnfdSAK2Imau12UM1mLluoedCBLTzzePE11kmd17JdUt7OCbGxHIHxnpkrDNOJQad3mOpWqI1hsfAAFbmBpq7.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/B/5/a4/1koN6ouNtKzS2OTaEkZQ0lZuTxo91j7N3i2BhJVAnfdSAK2Imau12UM1mLluoedCBLTzzePE11kmd17JdUt7OCbGxHIHxnpkrDNOJQad3mOpWqI1hsfAAFbmBpq7.jpg')
    ImageInfo(id=176024256, name='IMG_1836.jpeg', link=None, hash='YZKQrZH5C0rBvGGA3p7hjWahz3/pV09u5m30Bz8GeYs=', mime='image/jpeg', ext='jpeg', size=140222, width=1067, height=800, labels_count=3, dataset_id=384126, created_at='2020-09-30T11:38:08.516Z', updated_at='2020-09-30T11:38:08.516Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/j/w/RB/lVQXAmF8eBPTYTKSiovHLJNULzMsKjMdB8VB3e97BmEJywEuF5fvQXXcQd7rYV1RvjksKD5TFzEpubefzb7vduz0cZgjOIOzaPCJZjfdHijcqqiRk4IXEZfuni6O.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/j/w/RB/lVQXAmF8eBPTYTKSiovHLJNULzMsKjMdB8VB3e97BmEJywEuF5fvQXXcQd7rYV1RvjksKD5TFzEpubefzb7vduz0cZgjOIOzaPCJZjfdHijcqqiRk4IXEZfuni6O.jpg')

If connection to the site is not established or image with given ID not exist raise RetryError.

Get list of information for all images in the selected dataset ID:

    images_info = api.image.get_list(384126)
    for image_info in images_info:
        print(image_info)

    ImageInfo(id=176024255, name='IMG_0748.jpeg', link=None, hash='aiTmLMhrlyUU0KDVK3HHR8/nbaPILltENG/jYgj3cUM=', mime='image/jpeg', ext='jpeg', size=155790, width=1067, height=800, labels_count=3, dataset_id=384126, created_at='2020-09-30T11:38:08.516Z', updated_at='2020-10-21T06:21:52.926Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/t/t5/VMCn7mwooVoNAokqxvpNPVRcUW52B0zPp2CzSz1tQP6l0H5xAO8zX9iuT6CmAggtnjVJ0tjZ9taJ5ChiC9rvmz8plmIOSViFBIePEyslSFYmFpzWf7Rf4rN8iIXx.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/P/t/t5/VMCn7mwooVoNAokqxvpNPVRcUW52B0zPp2CzSz1tQP6l0H5xAO8zX9iuT6CmAggtnjVJ0tjZ9taJ5ChiC9rvmz8plmIOSViFBIePEyslSFYmFpzWf7Rf4rN8iIXx.jpg')
    ImageInfo(id=176024256, name='IMG_1836.jpeg', link=None, hash='YZKQrZH5C0rBvGGA3p7hjWahz3/pV09u5m30Bz8GeYs=', mime='image/jpeg', ext='jpeg', size=140222, width=1067, height=800, labels_count=3, dataset_id=384126, created_at='2020-09-30T11:38:08.516Z', updated_at='2020-09-30T11:38:08.516Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/j/w/RB/lVQXAmF8eBPTYTKSiovHLJNULzMsKjMdB8VB3e97BmEJywEuF5fvQXXcQd7rYV1RvjksKD5TFzEpubefzb7vduz0cZgjOIOzaPCJZjfdHijcqqiRk4IXEZfuni6O.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/j/w/RB/lVQXAmF8eBPTYTKSiovHLJNULzMsKjMdB8VB3e97BmEJywEuF5fvQXXcQd7rYV1RvjksKD5TFzEpubefzb7vduz0cZgjOIOzaPCJZjfdHijcqqiRk4IXEZfuni6O.jpg')
    ImageInfo(id=176024257, name='IMG_2084.jpeg', link=None, hash='0E27qU9e5gH3g6HJF0P9/mq/W57lJInX5n1Q9N0SWZY=', mime='image/jpeg', ext='jpeg', size=142097, width=1067, height=800, labels_count=7, dataset_id=384126, created_at='2020-09-30T11:38:08.516Z', updated_at='2020-09-30T11:38:08.516Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/t/d/XV/vekdUILPlrtJJ5zkuRoISJIxiGYfDCXm8tNy4sFIO8GYGXlB2VGtvOwT5xR2Mx14WfKbmp7BYAjbXk4RZXJsGEKqrYgBSGkhFhMEPszn8smtMG6kiFEySLsDrGO8.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/t/d/XV/vekdUILPlrtJJ5zkuRoISJIxiGYfDCXm8tNy4sFIO8GYGXlB2VGtvOwT5xR2Mx14WfKbmp7BYAjbXk4RZXJsGEKqrYgBSGkhFhMEPszn8smtMG6kiFEySLsDrGO8.jpg')

If connection to the site is not established or dataset with given ID not exist raise RetryError.

Download image from api by it ID:

    image = api.image.download_np(93907659)
    print(type(image))
    print(image.shape)

    <class 'numpy.ndarray'>
    (800, 1067, 3)

NOTE: received image have RGB format!

Download images with given IDs from dataset with given ID in numpy format:

    images = api.image.download_nps(384126, [193526437, 176024256])
    for image in images:
        print(type(image), image.shape)

    <class 'numpy.ndarray'> (800, 1067, 3)
    <class 'numpy.ndarray'> (800, 1067, 3)

Download image from API by it ID and save it on the host:

    api.image.download_path(93907659, os.path.join(os.getcwd(), image_info.name))

Download images from API with given IDs and dataset ID, saves them on the host:

    api.image.download_paths(384126, [193526437, 176024256], [os.path.join(os.getcwd(), 'image1.jpg', os.path.join(os.getcwd(), 'image2.jpg'))])

Upload image with given name from host to dataset on API:

    new_image_info = api.image.upload_path(384126, 'image_name', os.path.join(os.getcwd(), 'new_image.jpeg'))
    print(new_image_info)

    ImageInfo(id=193564574, name='image_name.jpeg', link=None, hash='UgoaTyUZq+z5rfVfTQN7spGY7zzUhcHuY+R8Ast1Uak=', mime='image/jpeg', ext='jpeg', size=214745, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:32:19.382Z', updated_at='2021-02-10T06:32:19.382Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/x/V/5r/hJoijrh1VCQIb8sG5sjTeBRwfe9idyJNNupjMZKNVImy2GQgHmqdhm5PPEO8QILrTl0l38olwKDFkDKrBb8BKLjkYfHVHTEuiMj60lWu4r1vlyu7XyEelofkFthS.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/x/V/5r/hJoijrh1VCQIb8sG5sjTeBRwfe9idyJNNupjMZKNVImy2GQgHmqdhm5PPEO8QILrTl0l38olwKDFkDKrBb8BKLjkYfHVHTEuiMj60lWu4r1vlyu7XyEelofkFthS.jpg')

Upload images with given names from host to dataset on API, if lengh of names list != lengh of paths list raise error:

    new_images_info = api.image.upload_paths(384126, ['image_name1', 'image_name2'], [os.path.join(os.getcwd(), 'new_image1.jpeg'), os.path.join(os.getcwd(), 'new_image2.jpeg')])
    for new_image_info in new_images_info:
        print(new_image_info)

    ImageInfo(id=193566094, name='image_name1.jpeg', link=None, hash='UgoaTyUZq+z5rfVfTQN7spGY7zzUhcHuY+R8Ast1Uak=', mime='image/jpeg', ext='jpeg', size=214745, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:36:05.895Z', updated_at='2021-02-10T06:36:05.895Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/x/V/5r/hJoijrh1VCQIb8sG5sjTeBRwfe9idyJNNupjMZKNVImy2GQgHmqdhm5PPEO8QILrTl0l38olwKDFkDKrBb8BKLjkYfHVHTEuiMj60lWu4r1vlyu7XyEelofkFthS.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/x/V/5r/hJoijrh1VCQIb8sG5sjTeBRwfe9idyJNNupjMZKNVImy2GQgHmqdhm5PPEO8QILrTl0l38olwKDFkDKrBb8BKLjkYfHVHTEuiMj60lWu4r1vlyu7XyEelofkFthS.jpg')
    ImageInfo(id=193566095, name='image_name2.jpeg', link=None, hash='UgoaTyUZq+z5rfVfTQN7spGY7zzUhcHuY+R8Ast1Uak=', mime='image/jpeg', ext='jpeg', size=214745, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:36:05.895Z', updated_at='2021-02-10T06:36:05.895Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/x/V/5r/hJoijrh1VCQIb8sG5sjTeBRwfe9idyJNNupjMZKNVImy2GQgHmqdhm5PPEO8QILrTl0l38olwKDFkDKrBb8BKLjkYfHVHTEuiMj60lWu4r1vlyu7XyEelofkFthS.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/x/V/5r/hJoijrh1VCQIb8sG5sjTeBRwfe9idyJNNupjMZKNVImy2GQgHmqdhm5PPEO8QILrTl0l38olwKDFkDKrBb8BKLjkYfHVHTEuiMj60lWu4r1vlyu7XyEelofkFthS.jpg')

Upload image in numpy with given name to dataset on API:

    image = api.image.download_np(93907659)
    new_image_info = api.image.upload_np(384126, 'image_name.jpg', image)
    print(new_image_info)

    ImageInfo(id=193566096, name='image_name.jpg', link=None, hash='ilKYs4pyP47YlihgiZIuVZU7/3MqaI4rJsCgfPpQqaM=', mime='image/jpeg', ext='jpeg', size=193755, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:37:25.433Z', updated_at='2021-02-10T06:37:25.433Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/B/5/a4/1koN6ouNtKzS2OTaEkZQ0lZuTxo91j7N3i2BhJVAnfdSAK2Imau12UM1mLluoedCBLTzzePE11kmd17JdUt7OCbGxHIHxnpkrDNOJQad3mOpWqI1hsfAAFbmBpq7.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/B/5/a4/1koN6ouNtKzS2OTaEkZQ0lZuTxo91j7N3i2BhJVAnfdSAK2Imau12UM1mLluoedCBLTzzePE11kmd17JdUt7OCbGxHIHxnpkrDNOJQad3mOpWqI1hsfAAFbmBpq7.jpg')

NOTE: if image name is specified without extension raise UnsupportedImageFormat error!

Upload images in numpy with given names to dataset on API, if lengh of names list != lengh of nps list raise error:

    image1 = api.image.download_np(93907659)
    image2 = api.image.download_np(93907660)
    new_images_info = api.image.upload_nps(384126, ['image_name1.jpg', 'image_name2.jpg'], [image1, image2])
    for new_image_info in new_images_info:
        print(new_image_info)

    ImageInfo(id=193566097, name='image_name1.jpg', link=None, hash='ilKYs4pyP47YlihgiZIuVZU7/3MqaI4rJsCgfPpQqaM=', mime='image/jpeg', ext='jpeg', size=193755, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:39:27.915Z', updated_at='2021-02-10T06:39:27.915Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/B/5/a4/1koN6ouNtKzS2OTaEkZQ0lZuTxo91j7N3i2BhJVAnfdSAK2Imau12UM1mLluoedCBLTzzePE11kmd17JdUt7OCbGxHIHxnpkrDNOJQad3mOpWqI1hsfAAFbmBpq7.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/B/5/a4/1koN6ouNtKzS2OTaEkZQ0lZuTxo91j7N3i2BhJVAnfdSAK2Imau12UM1mLluoedCBLTzzePE11kmd17JdUt7OCbGxHIHxnpkrDNOJQad3mOpWqI1hsfAAFbmBpq7.jpg')
    ImageInfo(id=193566098, name='image_name2.jpg', link=None, hash='mkwqMWs1x5s7XTxbKndcKWVzzhlAsbeklvjiOnCSPiw=', mime='image/jpeg', ext='jpeg', size=199703, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:39:27.915Z', updated_at='2021-02-10T06:39:27.915Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/d/T/uA/2iQ6Lijl66g1cjbApPxBY6x6afCatShWhb4Hg5PmkX4UclkWd2EIDeZnD5DwUa8RU7X50pJsI0LgU2Xg3jPy3SVmJIEhkLoDH1J8P1k3tTJ1uvsoFFfTdg4xpBnq.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/d/T/uA/2iQ6Lijl66g1cjbApPxBY6x6afCatShWhb4Hg5PmkX4UclkWd2EIDeZnD5DwUa8RU7X50pJsI0LgU2Xg3jPy3SVmJIEhkLoDH1J8P1k3tTJ1uvsoFFfTdg4xpBnq.jpg')

NOTE: if images names is specified without extension raise UnsupportedImageFormat error!

Upload image from given link with given name to dataset on API:

    link = 'https://m.media-amazon.com/images/M/MV5BMjA0MDIwMDYwNl5BMl5BanBnXkFtZTcwMjY0Mzg4Mw@@._V1_SY1000_CR0,0,1350,1000_AL_.jpg'
    new_image_info = api.image.upload_link(384126, 'image_name.jpg', link)
    print(new_image_info)

    ImageInfo(id=193567643, name='image_name.jpg', link='https://m.media-amazon.com/images/M/MV5BMjA0MDIwMDYwNl5BMl5BanBnXkFtZTcwMjY0Mzg4Mw@@._V1_SY1000_CR0,0,1350,1000_AL_.jpg', hash='7k1ftVchHNTlZ1ufzCRgULhNymsyeyGaWu/FtpifIKM=', mime='image/jpeg', ext='jpeg', size=101814, width=1350, height=1000, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:42:43.427Z', updated_at='2021-02-10T06:42:43.427Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/https://m.media-amazon.com/images/M/MV5BMjA0MDIwMDYwNl5BMl5BanBnXkFtZTcwMjY0Mzg4Mw@@._V1_SY1000_CR0,0,1350,1000_AL_.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/https://m.media-amazon.com/images/M/MV5BMjA0MDIwMDYwNl5BMl5BanBnXkFtZTcwMjY0Mzg4Mw@@._V1_SY1000_CR0,0,1350,1000_AL_.jpg')

Upload images from given links with given names to dataset on API, if lengh of names list != lengh of links list raise error:

    link1 = 'https://m.media-amazon.com/images/M/MV5BMTc5Njg5Njg2MV5BMl5BanBnXkFtZTgwMjAwMzg5MTE@._V1_SY1000_CR0,0,1332,1000_AL_.jpg'
    link2 = 'https://m.media-amazon.com/images/M/MV5BMjA5ODU3NTI0Ml5BMl5BanBnXkFtZTcwODczMTk2Mw@@._V1_SX1777_CR0,0,1777,756_AL_.jpg'
    new_images_info = api.image.upload_links(384126, ['image_name1.jpg', 'image_name2.jpg'], [link1, link2])
    for new_image_info in new_images_info:
        print(new_image_info)

    ImageInfo(id=193567644, name='image_name1.jpg', link='https://m.media-amazon.com/images/M/MV5BMTc5Njg5Njg2MV5BMl5BanBnXkFtZTgwMjAwMzg5MTE@._V1_SY1000_CR0,0,1332,1000_AL_.jpg', hash='mdW5Anc2uH7KSGEQoe0ov1VeDAuNy2h2EahHWm5Cxt0=', mime='image/jpeg', ext='jpeg', size=210307, width=1332, height=1000, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:44:27.687Z', updated_at='2021-02-10T06:44:27.687Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/https://m.media-amazon.com/images/M/MV5BMTc5Njg5Njg2MV5BMl5BanBnXkFtZTgwMjAwMzg5MTE@._V1_SY1000_CR0,0,1332,1000_AL_.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/https://m.media-amazon.com/images/M/MV5BMTc5Njg5Njg2MV5BMl5BanBnXkFtZTgwMjAwMzg5MTE@._V1_SY1000_CR0,0,1332,1000_AL_.jpg')
    ImageInfo(id=193567645, name='image_name2.jpg', link='https://m.media-amazon.com/images/M/MV5BMjA5ODU3NTI0Ml5BMl5BanBnXkFtZTcwODczMTk2Mw@@._V1_SX1777_CR0,0,1777,756_AL_.jpg', hash='pA3YGTC9H1j572xMfQgVDduNNDTHjJYhFVCM0rY7wS0=', mime='image/jpeg', ext='jpeg', size=147203, width=1777, height=756, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:44:27.687Z', updated_at='2021-02-10T06:44:27.687Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/https://m.media-amazon.com/images/M/MV5BMjA5ODU3NTI0Ml5BMl5BanBnXkFtZTcwODczMTk2Mw@@._V1_SX1777_CR0,0,1777,756_AL_.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/https://m.media-amazon.com/images/M/MV5BMjA5ODU3NTI0Ml5BMl5BanBnXkFtZTcwODczMTk2Mw@@._V1_SX1777_CR0,0,1777,756_AL_.jpg')

Upload image with given id with given name to dataset on API:

    new_image_id = 172627135
    new_image_info = api.image.upload_id(384126, 'image_name', new_image_id)
    print(new_image_info)

    ImageInfo(id=193567646, name='image_name.jpeg', link=None, hash='XnhZdcMaV4hkksiA47iWGalTuyg9d2vsLzfGLn2B/PM=', mime='image/jpeg', ext='jpeg', size=405245, width=1200, height=952, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:48:26.721Z', updated_at='2021-02-10T06:48:26.721Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/0/L/vH/WBkess4MaSyfMRPvO24eKAJ1KPPWlK4pvTiE2KkDQqeTi2lQkvB2nVTQ5ynVyVzYxlTCIaMTZzLrEVZ5OJsKIDM7hgcyEuNVxASCNDJ0smauj8rhbzfJjARWDY7n.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/0/L/vH/WBkess4MaSyfMRPvO24eKAJ1KPPWlK4pvTiE2KkDQqeTi2lQkvB2nVTQ5ynVyVzYxlTCIaMTZzLrEVZ5OJsKIDM7hgcyEuNVxASCNDJ0smauj8rhbzfJjARWDY7n.jpg')

Upload images with given ids with given names to dataset on API, if lengh of names list != lengh of ids list raise error::

    id1 = 172627139
    id2 = 172627142
    new_images_info = api.image.upload_ids(384126, ['image_name1', 'image_name2'], [id1, id2])
    for new_image_info in new_images_info:
        print(new_image_info)

    ImageInfo(id=193567647, name='image_name1.jpeg', link=None, hash='7qLZt8I9W6VR8YxtweN2nFQbctyHJLkU8TZqPzm4Hb4=', mime='image/jpeg', ext='jpeg', size=336919, width=1076, height=784, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:50:22.032Z', updated_at='2021-02-10T06:50:22.032Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/j/7/vy/hmPTGdlIUOH66HekSj7ScLriljsymFJdFZ3Fho3Ql518gRRE3a6ZDda39WM6RU8jafQgT1iP2XWRPI492abWsxlrNTIGbc6omawEGNibr8E2pd8jpuizihsWmO9M.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/j/7/vy/hmPTGdlIUOH66HekSj7ScLriljsymFJdFZ3Fho3Ql518gRRE3a6ZDda39WM6RU8jafQgT1iP2XWRPI492abWsxlrNTIGbc6omawEGNibr8E2pd8jpuizihsWmO9M.jpg')
    ImageInfo(id=193567648, name='image_name2.jpeg', link=None, hash='mwASdUTKE2xXuxqU1ekXK4JwqJDhOy74V0wKe4cvXss=', mime='image/jpeg', ext='jpeg', size=244613, width=980, height=589, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:50:22.032Z', updated_at='2021-02-10T06:50:22.032Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/a/W/jl/JVETAS1W3MhrD7UsLgei0n9WxE6Ch5CgLURXgHVRX9fkxOpEzUWKfXv0YHupurPH5FS5Lp1Lh33gQml7IhLgmQYULD1AJGX1U7q85cXZrdYQzhpdw1BLzeigQC9Z.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/a/W/jl/JVETAS1W3MhrD7UsLgei0n9WxE6Ch5CgLURXgHVRX9fkxOpEzUWKfXv0YHupurPH5FS5Lp1Lh33gQml7IhLgmQYULD1AJGX1U7q85cXZrdYQzhpdw1BLzeigQC9Z.jpg')

NOTE: In all cases of images uploading if image with given name already exists in dataset raise HTTPError!

Copy image with given id in destination dataset, if images with the same names already exist in destination dataset raise error, to avoid it, use change_name_if_conflict=True:

    new_image_info = api.image.copy(384126, 93907659, change_name_if_conflict=True)
    print(new_image_info)

    ImageInfo(id=193568050, name='IMG_6896.jpeg', link=None, hash='n3+8xGisUeHa6BF+ndaHdfH7Mm9XJaUTFbtPIZ5te7Y=', mime='image/jpeg', ext='jpeg', size=192651, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:54:32.653Z', updated_at='2021-02-10T06:54:32.653Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/r/O/kn/g101i1Dxb7tMoB804wudemjxMraoAjoQ8OpjtCItJUWqySTeItDERtiz0REAnW9IETqIE4gIgOiTM91UnK41dWIUubgwCM6b8JFnuIIU6MeEgFmabMrWxylfKkif.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/r/O/kn/g101i1Dxb7tMoB804wudemjxMraoAjoQ8OpjtCItJUWqySTeItDERtiz0REAnW9IETqIE4gIgOiTM91UnK41dWIUubgwCM6b8JFnuIIU6MeEgFmabMrWxylfKkif.jpg')

Copy images with given ids in destination dataset, if images with the same names already exist in destination dataset raise error, to avoid it, use change_name_if_conflict=True:

    new_images_info = api.image.copy_batch(384126, [93907659, 93907660], change_name_if_conflict=True)
    for new_image_info in new_images_info:
        print(new_image_info)

    ImageInfo(id=193568051, name='IMG_6896_01.jpeg', link=None, hash='n3+8xGisUeHa6BF+ndaHdfH7Mm9XJaUTFbtPIZ5te7Y=', mime='image/jpeg', ext='jpeg', size=192651, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:56:31.156Z', updated_at='2021-02-10T06:56:31.156Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/r/O/kn/g101i1Dxb7tMoB804wudemjxMraoAjoQ8OpjtCItJUWqySTeItDERtiz0REAnW9IETqIE4gIgOiTM91UnK41dWIUubgwCM6b8JFnuIIU6MeEgFmabMrWxylfKkif.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/r/O/kn/g101i1Dxb7tMoB804wudemjxMraoAjoQ8OpjtCItJUWqySTeItDERtiz0REAnW9IETqIE4gIgOiTM91UnK41dWIUubgwCM6b8JFnuIIU6MeEgFmabMrWxylfKkif.jpg')
    ImageInfo(id=193568052, name='IMG_8454.jpeg', link=None, hash='lPwQ304K4lro44mxRZFcanqZZfGzCPHRTEI60sf/brE=', mime='image/jpeg', ext='jpeg', size=198730, width=1067, height=800, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:56:31.156Z', updated_at='2021-02-10T06:56:31.156Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/i/x/wj/cNpxCQ99gbbDx7uA3Sg4ktHsATPMVStMCKwCQaXCDKi3MG9YXEShTywd1fXG0IT5DKGEAWfj8ong2EacpyhB7ckroETzAMOLbsOhebGk7UhR67WQq3NvUshV80RF.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/i/x/wj/cNpxCQ99gbbDx7uA3Sg4ktHsATPMVStMCKwCQaXCDKi3MG9YXEShTywd1fXG0IT5DKGEAWfj8ong2EacpyhB7ckroETzAMOLbsOhebGk7UhR67WQq3NvUshV80RF.jpg')

Move image with given id in destination dataset, if images with the same names already exist in destination dataset raise error, to avoid it, use change_name_if_conflict=True:

    new_image_info = api.image.move(384126, 172631635, change_name_if_conflict=True)
    print(new_image_info)

    ImageInfo(id=193568108, name='image_03_10.jpeg', link=None, hash='a7OYgj3ZY9wjss/sQlpftxGdqm+K0rihymQc9LCe5ls=', mime='image/jpeg', ext='jpeg', size=238139, width=734, height=639, labels_count=0, dataset_id=384126, created_at='2021-02-10T06:58:51.823Z', updated_at='2021-02-10T06:58:51.823Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/f/7/a7/7bFeNVEPW7dgzDO1QZiVDhGJLG8vAfVchVssRXmKNOAQJV3TXNsYDRxU5BepFncFwPxMbTnMt74dbhnq0jjcbl7Xoyx0rMaurg4Gj63rbFV0GtFuKaKpmYPjPRu1.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/f/7/a7/7bFeNVEPW7dgzDO1QZiVDhGJLG8vAfVchVssRXmKNOAQJV3TXNsYDRxU5BepFncFwPxMbTnMt74dbhnq0jjcbl7Xoyx0rMaurg4Gj63rbFV0GtFuKaKpmYPjPRu1.jpg')

Move images with given ids in destination dataset, if images with the same names already exist in destination dataset raise error, to avoid it, use change_name_if_conflict=True:

    new_images_info = api.image.move_batch(384126, [172631636, 172631627], change_name_if_conflict=True)
    for new_image_info in new_images_info:
        print(new_image_info)

    ImageInfo(id=193569319, name='image_03_11.jpeg', link=None, hash='QfZSwV7eR5XoqF2xV99X3nIeVe+5Sbw5vrd0UPXbDiA=', mime='image/jpeg', ext='jpeg', size=212421, width=793, height=549, labels_count=0, dataset_id=384126, created_at='2021-02-10T07:00:25.414Z', updated_at='2021-02-10T07:00:25.414Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/0/j/XW/BPfRrl6PZYcRrzKy2L5PAkTNt22OmZtK9orvSugK3Lfya4uwOEhOGq5fhSDOhRm7Rhs6EOtPxUYi3cPOecUlkD3CtkStSZiIugjNuyixZ0gVh8hjyNZRxQ24jrQl.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/0/j/XW/BPfRrl6PZYcRrzKy2L5PAkTNt22OmZtK9orvSugK3Lfya4uwOEhOGq5fhSDOhRm7Rhs6EOtPxUYi3cPOecUlkD3CtkStSZiIugjNuyixZ0gVh8hjyNZRxQ24jrQl.jpg')
    ImageInfo(id=193569320, name='image_03_02.jpeg', link=None, hash='LMIm6DkswlRu6Pf9HitKly8q8IyptACx9TclheAsoZk=', mime='image/jpeg', ext='jpeg', size=197685, width=729, height=525, labels_count=0, dataset_id=384126, created_at='2021-02-10T07:00:25.414Z', updated_at='2021-02-10T07:00:25.414Z', meta={}, path_original='/h5un6l2bnaz1vj8a9qgms4-public/images/original/a/K/cB/sSy82mXxOp3NAKVI1HyrSYsJqbZMPSrN7DD5T8SHvwadjnza4I7MAlB91l8Q7W41kCjCRNwetTlCNOi8Ljl9oqq3V1u10n5aVeloAr7jXj5m49DkuGrXCouCAneO.jpg', full_storage_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/a/K/cB/sSy82mXxOp3NAKVI1HyrSYsJqbZMPSrN7DD5T8SHvwadjnza4I7MAlB91l8Q7W41kCjCRNwetTlCNOi8Ljl9oqq3V1u10n5aVeloAr7jXj5m49DkuGrXCouCAneO.jpg')

NOTE: in all cases of coping and moving images you can use with_annotations=True flag to copy/move image with annotation.

Check validation of image extension:
    
    print(sly.image.is_valid_ext('.png'))
    print(sly.image.is_valid_ext('.py'))

    True
    False

Check validation of file extension:

    print(sly.image.has_valid_ext(os.path.join(os.getcwd(), 'new_image.jpeg')))
    print(sly.image.has_valid_ext(os.path.join(os.getcwd(), '016_image.py')))

    True
    False

Validate image extention, if image extention not supported raise ImageExtensionError:

    print(sly.image.validate_ext(os.path.join(os.getcwd(), 'new_image.jpeg')))
    try:
        print(sly.image.validate_ext(os.path.join(os.getcwd(), '016_image.py')))
    except ImageExtensionError as error:
        print(error)

    None
    Unsupported image extension: '.py' for file '/home/andrew/alex_work/016_image.py'. Only the following extensions are supported: .jpg, .jpeg, .mpo, .bmp, .png, .webp.

Validate input file format, if file extention not supported raise ImageExtensionError:

    print(sly.image.validate_format(os.path.join(os.getcwd(), 'new_image.jpeg')))
    try:
        print(sly.image.validate_format(os.path.join(os.getcwd(), '016_image.py')))
    except ImageReadException as error:
        print(error)

    None
    Error has occured trying to read image '/home/andrew/alex_work/016_image.py'. Original exception message: "cannot identify image file '/home/andrew/alex_work/016_image.py'"


Write image on host from numpy array:

    sly.image.write(os.path.join(os.getcwd(), 'write_image.jpeg'), image)

Compresses image and stores it in the byte object:

    image_bytes = sly.image.write_bytes(image, 'jpeg')
    print(image_bytes)
    print(type(image_bytes))

    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\...
    <class 'bytes'>

Get hash from input image:

    hash = sly.image.get_hash(image, 'jpeg')
    print(hash)

    9dTHLcAUfgVPgalcAF9UNHqRNvsF09RnBmqCE2DTgfQ=

Convert image to data url:

    url = sly.image.np_image_to_data_url(image)
    print(type(url))
    print(len(url))

    <class 'str'>
    1354226

Convert data url to numpy:

    mask = sly.image.data_url_to_numpy(url)
    print(type(mask))

    <class 'numpy.ndarray'>

## Images transformations

Start image for transformations(shape: (800, 1067, 3)):

![](https://i.imgur.com/BHUALdv.jpg)

Crop image with given rectangle:

    crop_image = sly.image.crop(image, sly.Rectangle(0, 0, 500, 600))

![](https://i.imgur.com/4tNm2GS.jpg)

If size of rectangle is more then image shape raise ValueError:

    try:
        crop_image = sly.image.crop(image, sly.Rectangle(0, 0, 5000, 6000))
    except ValueError as error:
        print(error)

    Rectangle for crop out of image area!

Crop cut out part of the image with rectangle size:

    crop_with_padding_image = sly.image.crop_with_padding(image, sly.Rectangle(0, 0, 1000, 1200))

![](https://i.imgur.com/Nv1UinH.jpg)

Resize image to given shape:

    resize_image = sly.image.resize(image, (300, 500))

![](https://i.imgur.com/Xya4yz0.jpg)

Resize image to match a certain size:

    resize_image_nearest = sly.image.resize_inter_nearest(image, (300, 700))

![](https://i.imgur.com/0O6yMDH.jpg)

Scale image:

    scale_image = sly.image.scale(image, 0.7)

![](https://i.imgur.com/JPwCbow.jpg)

Flip image from left to right:

    fliplr_image = sly.image.fliplr(image)

![](https://i.imgur.com/1mqnuZU.jpg)

Flip image from up to down:

    flipud_image = sly.image.flipud(image)

![](https://i.imgur.com/LDwRDvm.jpg)

Rotate image(keep_black mode):

    from supervisely_lib.imaging.image import RotateMode
    rotate_im_keep_black = sly.image.rotate(image, 45)

![](https://i.imgur.com/VjiwV4O.jpg)

Rotate image(crop_black mode):

    rotate_im_crop_black = sly.image.rotate(image, 45, RotateMode.CROP_BLACK)

![](https://i.imgur.com/Rs34eMa.jpg)

Rotate image(origin_size mode):

    rotate_im_origin_size = sly.image.rotate(image, 5, RotateMode.SAVE_ORIGINAL_SIZE) * 255

![](https://i.imgur.com/ttDWfBE.jpg)

## Images processing

Changes contrast of image randomly:

    rand_contrast_im = sly.image.random_contrast(image, 1.1, 1.8)

![](https://i.imgur.com/i0WaepI.jpg)

Changes brightness of image randomly:

    rand_brightness_im = sly.image.random_brightness(image, 1.5, 8.5)

![](https://i.imgur.com/bOYwwYH.jpg)

Add random noise to image:

    random_noise_im = sly.image.random_noise(image, 25, 19)

![](https://i.imgur.com/EzyEHeM.jpg)

Changes image colors by randomly scaling each of RGB components:

    random_color_scale_im = sly.image.random_color_scale(image, 0.5, 0.9)

![](https://i.imgur.com/GGUZqlA.jpg)

Blurs an image:

    blur_im = sly.image.blur(image, 7)

![](https://i.imgur.com/wFnBaC6.jpg)

Median blurs an image:

    median_blur_im = sly.image.median_blur(image, 5)

NOTE: kernel_size must be odd and greater than 1, for example: 3, 5, 7...

![](https://i.imgur.com/FQ977ON.jpg)

Blurs an image using a Gaussian filter:

    gaussian_blur_im = sly.image.gaussian_blur(image, 3.3, 7.5)

![](https://i.imgur.com/brs6Au0.jpg)

Draws text on image:

    sly.image.draw_text(image, 'you text', (100, 50))

![](https://i.imgur.com/iMObFor.jpg)

Draws text sequence on image:

    sly.image.draw_text_sequence(image, ['some_text', 'another_text'], (10, 10))

![](https://i.imgur.com/wIzrDuf.jpg)
