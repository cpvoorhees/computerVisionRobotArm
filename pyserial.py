import serial
import time

# arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1) 
# def write_read(width, height, depth): 
# 	arduino.write(bytes(width,height, depth 'utf-8')) 
# 	time.sleep(0.05) 
# 	data = arduino.readline() 
# 	return data 
def Timofey(width, height, depth):
    def Timofey(width, height, depth):
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

    serialInst.write(width, height,depth, 'utf-8' )

    time.sleep(5)