for d in data_hr:
    for i in os.listdir(d):
        if os.path.isfile(os.path.join(d, i)):
            os.path.join(d, i)