import numpy as np
import matplotlib.pyplot as plt
import logging as log

from .WpcCommon import *

def WpcAskPacketExtractor(data, index, freq, rssi):

	log.debug(f'Ask packet extractor is running with input data size = {data.size}')

	sync_bit  = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,1], dtype='bool')
	start_bit = np.array([0,0], dtype='bool')
	stop_bit  = np.array([1,0], dtype='bool')
	
	packet_detected = False
	dataPacket = np.empty(0, dtype='bool')
	packet = []
	meta_data = []

	i = 10
	while i<data.size:
		if packet_detected:
			if ( np.all(data[i:i+2] == start_bit) or np.all(data[i:i+2] == np.logical_not(start_bit)) ) and \
			( np.all(data[i+20:i+22] == stop_bit) or np.all(data[i+20:i+22] == np.logical_not(stop_bit)) ) and \
			( np.all(ManchesterDecoder(data[i-2:i+22])[1][1:] == 0) ) and \
			ParityCheck(ManchesterDecoder(data[i+2:i+20])[0]):
				dataPacket = np.append(dataPacket, ManchesterDecoder(data[i:i+22])[0])
				i += 22
			else: ## end of detected packet
				if dataPacket.size > 11:
					packet.append(dataPacket)
					meta_data.append(np.array([ index[i-2*dataPacket.size], index[i], 1, freq[index[i-2*dataPacket.size]], rssi[index[i-2*dataPacket.size]] ]))

				log.debug(f'ASK demod: End of data packet is detected -- len= {dataPacket.size} @{index[i]} ')
				dataPacket = np.empty(0, dtype='bool')
				packet_detected = False
				i += 1

		else:
			if ( np.all(data[i-(sync_bit.size-2):i+2] == sync_bit) or np.all(data[i-(sync_bit.size-2):i+2] == np.logical_not(sync_bit)) ) and \
			( np.all(data[i:i+2] == start_bit) or np.all(data[i:i+2] == np.logical_not(start_bit)) ) and \
			( np.all(data[i+20:i+22] == stop_bit) or np.all(data[i+20:i+22] == np.logical_not(stop_bit)) ) and \
			( np.all(ManchesterDecoder(data[i-2:i+22])[1][1:] == 0)) and \
			ParityCheck(ManchesterDecoder(data[i+2:i+20])[0]):
				log.debug(f'ASK demod: data packet is detected = {index[i]} ')
				dataPacket = np.append(dataPacket, ManchesterDecoder(data[i:i+22])[0])
				packet_detected = True
				i += 22
			else:
				i += 1

	return packet, meta_data

def WpcFskPacketExtractor(data, index, freq, rssi):
	
	ack_packet = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0], dtype='bool')
	nak_packet = np.array([1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0], dtype='bool')
	nd_packet  = np.array([1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0], dtype='bool')
	atn_packet = np.array([1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0], dtype='bool')

	start_bit  = np.array([0,0], dtype='bool')
	stop_bit   = np.array([1,0], dtype='bool')

	dataPacket = np.empty(0, dtype='bool')
	packet = []
	meta_data = []

	data = data.astype('bool')
	log.debug(f'Fsk packet extractor is running with input data size = {data.size}')

	packet_detected = False
	i = 0
	while i<data.size:
		if packet_detected:
			if ( np.all(data[i:i+2] == start_bit) or np.all(data[i:i+2] == np.logical_not(start_bit)) ) and \
			( np.all(data[i+20:i+22] == stop_bit) or np.all(data[i+20:i+22] == np.logical_not(stop_bit)) )and \
			np.all(ManchesterDecoder(data[i:i+22])[1][1:] == False) and \
			not(ParityCheck(ManchesterDecoder(data[i+2:i+20])[0])):
			
				dataPacket = np.append(dataPacket, ManchesterDecoder(data[i:i+22])[0])
				i += 22
				i = data.size-1 if i>=data.size else i
			else:
				if dataPacket.size > 11:
					packet.append(dataPacket)
					meta_data.append(np.array([ index[i-2*dataPacket.size], index[i], 0, freq[index[i-2*dataPacket.size]], rssi[index[i-2*dataPacket.size]] ]))
					log.debug(f'FSK demod: End of data packet is detected -- len= {dataPacket.size} @{index[i]} ')
					i += 1
				else:
					## some ack/nak detected as data packet wrongly, with this condition try to ignore it
					log.debug(f'FSK demod: data packet is ignored -- len less than 2 bytes({dataPacket.size}) @{index[i]} ')
					i -= 20
				
				dataPacket = np.empty(0, dtype='bool')
				packet_detected = False
		else:
			if np.all(data[i:i+16] == ack_packet) or np.all(data[i:i+16] == np.logical_not(ack_packet)):
				log.debug(f'FSK demod: ACK packet is detected = {index[i]} ')
				packet.append(ManchesterDecoder(ack_packet)[0])
				meta_data.append(np.array([ index[i], index[i+15], 0, freq[index[i]], rssi[index[i]] ]))
				i += 16
			elif np.all(data[i:i+16] == nak_packet) or np.all(data[i:i+16] == np.logical_not(nak_packet)):
				log.debug(f'FSK demod: NAK packet is detected = {index[i]} ')
				packet.append(ManchesterDecoder(nak_packet)[0])
				meta_data.append(np.array([ index[i], index[i+15], 0, freq[index[i]], rssi[index[i]] ]))
				i += 16
			elif np.all(data[i:i+16] == nd_packet) or np.all(data[i:i+16] == np.logical_not(nd_packet)):
				log.debug(f'FSK demod: ND packet is detected = {index[i]} ')
				packet.append(ManchesterDecoder(nd_packet)[0])
				meta_data.append(np.array([ index[i], index[i+15], 0, freq[index[i]], rssi[index[i]] ]))
				i += 16
			elif np.all(data[i:i+16] == atn_packet) or np.all(data[i:i+16] == np.logical_not(atn_packet)):
				log.debug(f'FSK demod: ATN packet is detected = {index[i]} ')
				packet.append(ManchesterDecoder(atn_packet)[0])
				meta_data.append(np.array([ index[i], index[i+15], 0, freq[index[i]], rssi[index[i]] ]))
				i += 16
			elif ( np.all(data[i:i+2] == start_bit) or np.all(data[i:i+2] == np.logical_not(start_bit)) ) and \
			( np.all(data[i+20:i+22] == stop_bit) or np.all(data[i+20:i+22] == np.logical_not(stop_bit)) ) and \
			np.all(ManchesterDecoder(data[i:i+22])[1][1:] == False) and \
			not(ParityCheck(ManchesterDecoder(data[i+2:i+20])[0])):
			
				log.debug(f'FSK demod: data packet is detected = {index[i]} ')
				dataPacket = np.append(dataPacket, ManchesterDecoder(data[i:i+22])[0])
				packet_detected = True
				i += 22
			else:
				i += 1

	#return packet, np.array(meta_data)
	return packet, meta_data

def WpcEventExtraction(freq, rssi, fs):
	data = np.array([1 if f > 10000 else 0 for f in freq])

	packets = []
	meta_data = []
	for i in range(1, data.size):
		if (data[i] != data[i-1]):
			packets.append(np.array([0x02, 0x33] if data[i] else [0x01, 0xcc]))
			meta_data.append(np.array([i, 0, 2, freq[i], rssi[i]]))
			#print('event packet detected:', freq[i], i)
			
	return packets, meta_data

def WpcPacket(ask, ask_index, fsk, fsk_index, freq, rssi, filename):
	packet_ask, ask_meta_data = WpcAskPacketExtractor(ask, ask_index, freq, rssi)
	packet_fsk, fsk_meta_data = WpcFskPacketExtractor(fsk, fsk_index, freq, rssi)
	packet_event, event_meta_data = WpcEventExtraction(freq, rssi, 400.0e3)
	
	packet = []
	for pkt in packet_ask:
		packet.append(pkt)
	for pkt in packet_fsk:
		packet.append(pkt)
	for pkt in packet_event:
		packet.append(pkt)
	
	meta_data = []
	for pkt in ask_meta_data:
		meta_data.append(pkt)
	for pkt in fsk_meta_data:
		meta_data.append(pkt)
	for pkt in event_meta_data:
		meta_data.append(pkt)
	meta_data = np.array(meta_data)
	
	outfile = open(filename, "w")
	type_dict = {0:'fsk', 1:'ask', 2:'event'}
	if meta_data.size == 0:
		log.info('there is no detected packet.')
	else:
		for i in np.argsort(meta_data.T[0]):
			message = f'{{ \"time\":{int(meta_data[i][0])}, \"modulation\":\"{type_dict[meta_data[i][2]]}\", \"freq\":\"{meta_data[i][3]:.2f}\", \"rssi\":\"{meta_data[i][4]:.2f}\", \"data\":{[val for val in packet[i].astype(int)]} }}'
			log.info(message)
			outfile.writelines(message)
			outfile.writelines('\n')
	outfile.close()
