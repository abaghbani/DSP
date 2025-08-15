import serial

def tx_rx_hci_reset():

    port = 'COM24'
    baudrate = 115200

    hci_reset = bytes([0x01, 0x03, 0x0C, 0x00])

    with serial.Serial(port, baudrate, timeout=2, rtscts=False, xonxoff=False, ) as ser:
        print(ser.name)  
        ser.write(hci_reset)
        print("Sent HCI Reset")
        response = ser.read(64)  # Read response (may need to adjust size/timeout)
        print("Response:", response.hex())
        ser.close()

