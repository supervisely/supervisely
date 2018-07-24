def load_config(file_path):
    with open(file_path) as f:
        config = f.read()
    return config


def refact_yolo_config(config, input_size_wh, batch_size, subdivisions, num_classes, lr):
    config = config.replace('cust_width', str(input_size_wh[0]))
    config = config.replace('cust_height', str(input_size_wh[1]))
    config = config.replace('cust_batch_size', str(batch_size))
    config = config.replace('cust_subdivisions', str(subdivisions))
    config = config.replace('cust_num_classes', str(num_classes))
    config = config.replace('cust_last_filters', str((num_classes + 5) * 3))
    config = config.replace('cust_lr', str(lr))

    return config


def save_config(config, file_path):
    with open(file_path, 'w') as f:
        f.write(config)
