import evdev

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
found_path = ""
for device in devices:
   print(device.path, device.name, device.phys)
   if device.name == "Logitech MX Master 2S":
       found_path = device.path
       break

if found_path:
    device = evdev.InputDevice(found_path)
    for event in device.read_loop():
        print(event)
        '''
        type 1 (EV_KEY), code 272  (['BTN_LEFT', 'BTN_MOUSE']), value 0
        type 1 (EV_KEY), code 273  (BTN_RIGHT), value 0
        type 2 (EV_REL), code 11   (REL_WHEEL_HI_RES), value 15
        type 2 (EV_REL), code 11   (REL_WHEEL_HI_RES), value -15

        type 1 (EV_KEY), code 275  (BTN_EXTRA), value 1
        type 1 (EV_KEY), code 276  (BTN_EXTRA), value 0
        '''




