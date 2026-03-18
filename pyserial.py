import serial
import time

def timofey(x,y,z):
    ports = serial.tools.list_ports.comports()
    serialInst = serial.Serial()
    portsList = []
    for onePort in ports:
        portsList.append(str(onePort))
        print(str(onePort))

    val = input("Select Port: COM")

    for x in range(0, len(portsList)):
        if portsList[x].startswith("COM" + str(val)):
            porVar = "COM" + str(val)
            print(porVar)
    serialInst.baudrate = 9600
    serialInst.port = porVar
    serialInst.open()

    while True:
        command = input("Arduino Command: (ON/OFF):")
        serialInst.write(bytes(x,y,z, 'utf-8'))
        time.sleep(1)

        exit()
  