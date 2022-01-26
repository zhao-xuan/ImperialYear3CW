
mttkrp:     file format elf64-littleaarch64


Disassembly of section .init:

0000000000001138 <_init>:
    1138:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    113c:	910003fd 	mov	x29, sp
    1140:	940000c7 	bl	145c <call_weak_fn>
    1144:	a8c17bfd 	ldp	x29, x30, [sp], #16
    1148:	d65f03c0 	ret

Disassembly of section .plt:

0000000000001150 <.plt>:
    1150:	a9bf7bf0 	stp	x16, x30, [sp, #-16]!
    1154:	b00000b0 	adrp	x16, 16000 <__FRAME_END__+0x100e8>
    1158:	f947fe11 	ldr	x17, [x16, #4088]
    115c:	913fe210 	add	x16, x16, #0xff8
    1160:	d61f0220 	br	x17
    1164:	d503201f 	nop
    1168:	d503201f 	nop
    116c:	d503201f 	nop

0000000000001170 <memcpy@plt>:
    1170:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1174:	f9400211 	ldr	x17, [x16]
    1178:	91000210 	add	x16, x16, #0x0
    117c:	d61f0220 	br	x17

0000000000001180 <exit@plt>:
    1180:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1184:	f9400611 	ldr	x17, [x16, #8]
    1188:	91002210 	add	x16, x16, #0x8
    118c:	d61f0220 	br	x17

0000000000001190 <__cxa_finalize@plt>:
    1190:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1194:	f9400a11 	ldr	x17, [x16, #16]
    1198:	91004210 	add	x16, x16, #0x10
    119c:	d61f0220 	br	x17

00000000000011a0 <fputc@plt>:
    11a0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11a4:	f9400e11 	ldr	x17, [x16, #24]
    11a8:	91006210 	add	x16, x16, #0x18
    11ac:	d61f0220 	br	x17

00000000000011b0 <clock_gettime@plt>:
    11b0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11b4:	f9401211 	ldr	x17, [x16, #32]
    11b8:	91008210 	add	x16, x16, #0x20
    11bc:	d61f0220 	br	x17

00000000000011c0 <asprintf@plt>:
    11c0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11c4:	f9401611 	ldr	x17, [x16, #40]
    11c8:	9100a210 	add	x16, x16, #0x28
    11cc:	d61f0220 	br	x17

00000000000011d0 <pow@plt>:
    11d0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11d4:	f9401a11 	ldr	x17, [x16, #48]
    11d8:	9100c210 	add	x16, x16, #0x30
    11dc:	d61f0220 	br	x17

00000000000011e0 <omp_get_thread_num@plt>:
    11e0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11e4:	f9401e11 	ldr	x17, [x16, #56]
    11e8:	9100e210 	add	x16, x16, #0x38
    11ec:	d61f0220 	br	x17

00000000000011f0 <fclose@plt>:
    11f0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11f4:	f9402211 	ldr	x17, [x16, #64]
    11f8:	91010210 	add	x16, x16, #0x40
    11fc:	d61f0220 	br	x17

0000000000001200 <fopen@plt>:
    1200:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1204:	f9402611 	ldr	x17, [x16, #72]
    1208:	91012210 	add	x16, x16, #0x48
    120c:	d61f0220 	br	x17

0000000000001210 <time@plt>:
    1210:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1214:	f9402a11 	ldr	x17, [x16, #80]
    1218:	91014210 	add	x16, x16, #0x50
    121c:	d61f0220 	br	x17

0000000000001220 <malloc@plt>:
    1220:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1224:	f9402e11 	ldr	x17, [x16, #88]
    1228:	91016210 	add	x16, x16, #0x58
    122c:	d61f0220 	br	x17

0000000000001230 <__isoc99_fscanf@plt>:
    1230:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1234:	f9403211 	ldr	x17, [x16, #96]
    1238:	91018210 	add	x16, x16, #0x60
    123c:	d61f0220 	br	x17

0000000000001240 <__libc_start_main@plt>:
    1240:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1244:	f9403611 	ldr	x17, [x16, #104]
    1248:	9101a210 	add	x16, x16, #0x68
    124c:	d61f0220 	br	x17

0000000000001250 <fgetc@plt>:
    1250:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1254:	f9403a11 	ldr	x17, [x16, #112]
    1258:	9101c210 	add	x16, x16, #0x70
    125c:	d61f0220 	br	x17

0000000000001260 <aligned_alloc@plt>:
    1260:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1264:	f9403e11 	ldr	x17, [x16, #120]
    1268:	9101e210 	add	x16, x16, #0x78
    126c:	d61f0220 	br	x17

0000000000001270 <memset@plt>:
    1270:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1274:	f9404211 	ldr	x17, [x16, #128]
    1278:	91020210 	add	x16, x16, #0x80
    127c:	d61f0220 	br	x17

0000000000001280 <realloc@plt>:
    1280:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1284:	f9404611 	ldr	x17, [x16, #136]
    1288:	91022210 	add	x16, x16, #0x88
    128c:	d61f0220 	br	x17

0000000000001290 <rand@plt>:
    1290:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1294:	f9404a11 	ldr	x17, [x16, #144]
    1298:	91024210 	add	x16, x16, #0x90
    129c:	d61f0220 	br	x17

00000000000012a0 <strdup@plt>:
    12a0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12a4:	f9404e11 	ldr	x17, [x16, #152]
    12a8:	91026210 	add	x16, x16, #0x98
    12ac:	d61f0220 	br	x17

00000000000012b0 <omp_get_num_threads@plt>:
    12b0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12b4:	f9405211 	ldr	x17, [x16, #160]
    12b8:	91028210 	add	x16, x16, #0xa0
    12bc:	d61f0220 	br	x17

00000000000012c0 <strerror@plt>:
    12c0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12c4:	f9405611 	ldr	x17, [x16, #168]
    12c8:	9102a210 	add	x16, x16, #0xa8
    12cc:	d61f0220 	br	x17

00000000000012d0 <strrchr@plt>:
    12d0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12d4:	f9405a11 	ldr	x17, [x16, #176]
    12d8:	9102c210 	add	x16, x16, #0xb0
    12dc:	d61f0220 	br	x17

00000000000012e0 <__gmon_start__@plt>:
    12e0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12e4:	f9405e11 	ldr	x17, [x16, #184]
    12e8:	9102e210 	add	x16, x16, #0xb8
    12ec:	d61f0220 	br	x17

00000000000012f0 <__getauxval@plt>:
    12f0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12f4:	f9406211 	ldr	x17, [x16, #192]
    12f8:	91030210 	add	x16, x16, #0xc0
    12fc:	d61f0220 	br	x17

0000000000001300 <abort@plt>:
    1300:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1304:	f9406611 	ldr	x17, [x16, #200]
    1308:	91032210 	add	x16, x16, #0xc8
    130c:	d61f0220 	br	x17

0000000000001310 <puts@plt>:
    1310:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1314:	f9406a11 	ldr	x17, [x16, #208]
    1318:	91034210 	add	x16, x16, #0xd0
    131c:	d61f0220 	br	x17

0000000000001320 <getopt_long@plt>:
    1320:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1324:	f9406e11 	ldr	x17, [x16, #216]
    1328:	91036210 	add	x16, x16, #0xd8
    132c:	d61f0220 	br	x17

0000000000001330 <strcmp@plt>:
    1330:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1334:	f9407211 	ldr	x17, [x16, #224]
    1338:	91038210 	add	x16, x16, #0xe0
    133c:	d61f0220 	br	x17

0000000000001340 <fread@plt>:
    1340:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1344:	f9407611 	ldr	x17, [x16, #232]
    1348:	9103a210 	add	x16, x16, #0xe8
    134c:	d61f0220 	br	x17

0000000000001350 <free@plt>:
    1350:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1354:	f9407a11 	ldr	x17, [x16, #240]
    1358:	9103c210 	add	x16, x16, #0xf0
    135c:	d61f0220 	br	x17

0000000000001360 <fwrite@plt>:
    1360:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1364:	f9407e11 	ldr	x17, [x16, #248]
    1368:	9103e210 	add	x16, x16, #0xf8
    136c:	d61f0220 	br	x17

0000000000001370 <fflush@plt>:
    1370:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1374:	f9408211 	ldr	x17, [x16, #256]
    1378:	91040210 	add	x16, x16, #0x100
    137c:	d61f0220 	br	x17

0000000000001380 <strcpy@plt>:
    1380:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1384:	f9408611 	ldr	x17, [x16, #264]
    1388:	91042210 	add	x16, x16, #0x108
    138c:	d61f0220 	br	x17

0000000000001390 <srand@plt>:
    1390:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1394:	f9408a11 	ldr	x17, [x16, #272]
    1398:	91044210 	add	x16, x16, #0x110
    139c:	d61f0220 	br	x17

00000000000013a0 <__isoc99_sscanf@plt>:
    13a0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13a4:	f9408e11 	ldr	x17, [x16, #280]
    13a8:	91046210 	add	x16, x16, #0x118
    13ac:	d61f0220 	br	x17

00000000000013b0 <printf@plt>:
    13b0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13b4:	f9409211 	ldr	x17, [x16, #288]
    13b8:	91048210 	add	x16, x16, #0x120
    13bc:	d61f0220 	br	x17

00000000000013c0 <__errno_location@plt>:
    13c0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13c4:	f9409611 	ldr	x17, [x16, #296]
    13c8:	9104a210 	add	x16, x16, #0x128
    13cc:	d61f0220 	br	x17

00000000000013d0 <GOMP_parallel@plt>:
    13d0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13d4:	f9409a11 	ldr	x17, [x16, #304]
    13d8:	9104c210 	add	x16, x16, #0x130
    13dc:	d61f0220 	br	x17

00000000000013e0 <putchar@plt>:
    13e0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13e4:	f9409e11 	ldr	x17, [x16, #312]
    13e8:	9104e210 	add	x16, x16, #0x138
    13ec:	d61f0220 	br	x17

00000000000013f0 <fprintf@plt>:
    13f0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13f4:	f940a211 	ldr	x17, [x16, #320]
    13f8:	91050210 	add	x16, x16, #0x140
    13fc:	d61f0220 	br	x17

Disassembly of section .text:

0000000000001400 <init_have_lse_atomics>:
    1400:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    1404:	d2800200 	mov	x0, #0x10                  	// #16
    1408:	910003fd 	mov	x29, sp
    140c:	97ffffb9 	bl	12f0 <__getauxval@plt>
    1410:	53082000 	ubfx	w0, w0, #8, #1
    1414:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1418:	a8c17bfd 	ldp	x29, x30, [sp], #16
    141c:	390a8420 	strb	w0, [x1, #673]
    1420:	d65f03c0 	ret

0000000000001424 <_start>:
    1424:	d280001d 	mov	x29, #0x0                   	// #0
    1428:	d280001e 	mov	x30, #0x0                   	// #0
    142c:	aa0003e5 	mov	x5, x0
    1430:	f94003e1 	ldr	x1, [sp]
    1434:	910023e2 	add	x2, sp, #0x8
    1438:	910003e6 	mov	x6, sp
    143c:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    1440:	f947ec00 	ldr	x0, [x0, #4056]
    1444:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0x100e8>
    1448:	f947e863 	ldr	x3, [x3, #4048]
    144c:	b00000a4 	adrp	x4, 16000 <__FRAME_END__+0x100e8>
    1450:	f947cc84 	ldr	x4, [x4, #3992]
    1454:	97ffff7b 	bl	1240 <__libc_start_main@plt>
    1458:	97ffffaa 	bl	1300 <abort@plt>

000000000000145c <call_weak_fn>:
    145c:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    1460:	f947e400 	ldr	x0, [x0, #4040]
    1464:	b4000040 	cbz	x0, 146c <call_weak_fn+0x10>
    1468:	17ffff9e 	b	12e0 <__gmon_start__@plt>
    146c:	d65f03c0 	ret

0000000000001470 <deregister_tm_clones>:
    1470:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1474:	910a8000 	add	x0, x0, #0x2a0
    1478:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    147c:	910a8021 	add	x1, x1, #0x2a0
    1480:	eb00003f 	cmp	x1, x0
    1484:	540000c0 	b.eq	149c <deregister_tm_clones+0x2c>  // b.none
    1488:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0x100e8>
    148c:	f947d021 	ldr	x1, [x1, #4000]
    1490:	b4000061 	cbz	x1, 149c <deregister_tm_clones+0x2c>
    1494:	aa0103f0 	mov	x16, x1
    1498:	d61f0200 	br	x16
    149c:	d65f03c0 	ret

00000000000014a0 <register_tm_clones>:
    14a0:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    14a4:	910a8000 	add	x0, x0, #0x2a0
    14a8:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    14ac:	910a8021 	add	x1, x1, #0x2a0
    14b0:	cb000021 	sub	x1, x1, x0
    14b4:	d37ffc22 	lsr	x2, x1, #63
    14b8:	8b810c41 	add	x1, x2, x1, asr #3
    14bc:	9341fc21 	asr	x1, x1, #1
    14c0:	b40000c1 	cbz	x1, 14d8 <register_tm_clones+0x38>
    14c4:	b00000a2 	adrp	x2, 16000 <__FRAME_END__+0x100e8>
    14c8:	f947f042 	ldr	x2, [x2, #4064]
    14cc:	b4000062 	cbz	x2, 14d8 <register_tm_clones+0x38>
    14d0:	aa0203f0 	mov	x16, x2
    14d4:	d61f0200 	br	x16
    14d8:	d65f03c0 	ret
    14dc:	d503201f 	nop

00000000000014e0 <__do_global_dtors_aux>:
    14e0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    14e4:	910003fd 	mov	x29, sp
    14e8:	f9000bf3 	str	x19, [sp, #16]
    14ec:	d00000b3 	adrp	x19, 17000 <memcpy@GLIBC_2.17>
    14f0:	394a8260 	ldrb	w0, [x19, #672]
    14f4:	35000140 	cbnz	w0, 151c <__do_global_dtors_aux+0x3c>
    14f8:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    14fc:	f947d400 	ldr	x0, [x0, #4008]
    1500:	b4000080 	cbz	x0, 1510 <__do_global_dtors_aux+0x30>
    1504:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1508:	f940a800 	ldr	x0, [x0, #336]
    150c:	97ffff21 	bl	1190 <__cxa_finalize@plt>
    1510:	97ffffd8 	bl	1470 <deregister_tm_clones>
    1514:	52800020 	mov	w0, #0x1                   	// #1
    1518:	390a8260 	strb	w0, [x19, #672]
    151c:	f9400bf3 	ldr	x19, [sp, #16]
    1520:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1524:	d65f03c0 	ret
    1528:	d503201f 	nop
    152c:	d503201f 	nop

0000000000001530 <frame_dummy>:
    1530:	17ffffdc 	b	14a0 <register_tm_clones>

0000000000001534 <main._omp_fn.0>:
	if(dev_id == -2) {
		nthreads = 1;
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
	} else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
		#pragma omp parallel
    1534:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1538:	910003fd 	mov	x29, sp
    153c:	f9000bf3 	str	x19, [sp, #16]
    1540:	aa0003f3 	mov	x19, x0
        {
            nthreads = omp_get_num_threads();
    1544:	97ffff5b 	bl	12b0 <omp_get_num_threads@plt>
    1548:	b9000260 	str	w0, [x19]
		#pragma omp parallel
    154c:	f9400bf3 	ldr	x19, [sp, #16]
    1550:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1554:	d65f03c0 	ret

0000000000001558 <print_usage>:
static void print_usage(char ** argv) {
    1558:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    155c:	910003fd 	mov	x29, sp
	printf("Usage: %s [options] \n\n", argv[0]);
    1560:	f9400001 	ldr	x1, [x0]
    1564:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1568:	91214000 	add	x0, x0, #0x850
    156c:	97ffff91 	bl	13b0 <printf@plt>
	printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    1570:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1574:	9121a000 	add	x0, x0, #0x868
    1578:	97ffff66 	bl	1310 <puts@plt>
	printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    157c:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1580:	91226000 	add	x0, x0, #0x898
    1584:	97ffff63 	bl	1310 <puts@plt>
	printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    1588:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    158c:	91234000 	add	x0, x0, #0x8d0
    1590:	97ffff60 	bl	1310 <puts@plt>
	printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    1594:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1598:	9124e000 	add	x0, x0, #0x938
    159c:	97ffff5d 	bl	1310 <puts@plt>
	printf("         -r RANK (the number of matrix columns, 16:default)\n");
    15a0:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    15a4:	91262000 	add	x0, x0, #0x988
    15a8:	97ffff5a 	bl	1310 <puts@plt>
	printf("         -v VALIDATION, --validate=VALIDFILE (a previous output file to compare against). This also removes randomisation from matrix creation\n");
    15ac:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    15b0:	91272000 	add	x0, x0, #0x9c8
    15b4:	97ffff57 	bl	1310 <puts@plt>
	printf("         --help\n");
    15b8:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    15bc:	91296000 	add	x0, x0, #0xa58
    15c0:	97ffff54 	bl	1310 <puts@plt>
	printf("\n");
    15c4:	52800140 	mov	w0, #0xa                   	// #10
    15c8:	97ffff86 	bl	13e0 <putchar@plt>
}
    15cc:	a8c17bfd 	ldp	x29, x30, [sp], #16
    15d0:	d65f03c0 	ret

00000000000015d4 <compareFile>:
}



int compareFile(FILE * fPtr1, FILE * fPtr2)
{
    15d4:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    15d8:	910003fd 	mov	x29, sp
    15dc:	a90153f3 	stp	x19, x20, [sp, #16]
    15e0:	f90013f5 	str	x21, [sp, #32]
    15e4:	aa0003f5 	mov	x21, x0
    15e8:	aa0103f4 	mov	x20, x1
	int ch1, ch2;
	do {
		ch1 = fgetc(fPtr1);
    15ec:	aa1503e0 	mov	x0, x21
    15f0:	97ffff18 	bl	1250 <fgetc@plt>
    15f4:	2a0003f3 	mov	w19, w0
		ch2 = fgetc(fPtr2);
    15f8:	aa1403e0 	mov	x0, x20
    15fc:	97ffff15 	bl	1250 <fgetc@plt>
		if (ch1 != ch2) {
    1600:	6b00027f 	cmp	w19, w0
    1604:	54000101 	b.ne	1624 <compareFile+0x50>  // b.any
			return -1;
		}
	} while (ch1 != EOF && ch2 != EOF);
    1608:	3100067f 	cmn	w19, #0x1
    160c:	1a9f07e2 	cset	w2, ne  // ne = any
    1610:	7100005f 	cmp	w2, #0x0
    1614:	3a411804 	ccmn	w0, #0x1, #0x4, ne  // ne = any
    1618:	54fffea1 	b.ne	15ec <compareFile+0x18>  // b.any

	/* If both files have reached end */
	if (ch1 == EOF && ch2 == EOF)
    161c:	4b0203e0 	neg	w0, w2
    1620:	14000002 	b	1628 <compareFile+0x54>
			return -1;
    1624:	12800000 	mov	w0, #0xffffffff            	// #-1
		return 0;
	else
		return -1;
}
    1628:	a94153f3 	ldp	x19, x20, [sp, #16]
    162c:	f94013f5 	ldr	x21, [sp, #32]
    1630:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1634:	d65f03c0 	ret

0000000000001638 <main>:
{
    1638:	d13283ff 	sub	sp, sp, #0xca0
    163c:	a9007bfd 	stp	x29, x30, [sp]
    1640:	910003fd 	mov	x29, sp
    1644:	a90153f3 	stp	x19, x20, [sp, #16]
    1648:	a9025bf5 	stp	x21, x22, [sp, #32]
    164c:	a90363f7 	stp	x23, x24, [sp, #48]
    1650:	a9046bf9 	stp	x25, x26, [sp, #64]
    1654:	a90573fb 	stp	x27, x28, [sp, #80]
    1658:	fd0033e8 	str	d8, [sp, #96]
    165c:	2a0003f3 	mov	w19, w0
    1660:	aa0103f4 	mov	x20, x1
	sptIndex mode = 0;
    1664:	b9009fff 	str	wzr, [sp, #156]
	sptIndex R = 16;
    1668:	52800200 	mov	w0, #0x10                  	// #16
    166c:	b9009be0 	str	w0, [sp, #152]
	int dev_id = -2;
    1670:	12800020 	mov	w0, #0xfffffffe            	// #-2
    1674:	b90097e0 	str	w0, [sp, #148]
	printf("niters: %d\n", niters);
    1678:	528000a1 	mov	w1, #0x5                   	// #5
    167c:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1680:	9129a000 	add	x0, x0, #0xa68
    1684:	97ffff4b 	bl	13b0 <printf@plt>
	if(argc <= 3) { // #Required arguments
    1688:	71000e7f 	cmp	w19, #0x3
    168c:	5400016d 	b.le	16b8 <main+0x80>
	bool random = true;
    1690:	52800035 	mov	w21, #0x1                   	// #1
	FILE *fo = NULL;
    1694:	d2800016 	mov	x22, #0x0                   	// #0
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    1698:	910213f9 	add	x25, sp, #0x84
    169c:	d00000b8 	adrp	x24, 17000 <memcpy@GLIBC_2.17>
    16a0:	91056318 	add	x24, x24, #0x158
    16a4:	f0000017 	adrp	x23, 4000 <sptOmpMTTKRP+0x13c>
    16a8:	9129e2f7 	add	x23, x23, #0xa78
				fo = fopen(optarg, "w");
    16ac:	9000003a 	adrp	x26, 5000 <_IO_stdin_used+0x7b8>
    16b0:	9104a35a 	add	x26, x26, #0x128
    16b4:	1400003e 	b	17ac <main+0x174>
		print_usage(argv);
    16b8:	aa1403e0 	mov	x0, x20
    16bc:	97ffffa7 	bl	1558 <print_usage>
		exit(1);
    16c0:	52800020 	mov	w0, #0x1                   	// #1
    16c4:	97fffeaf 	bl	1180 <exit@plt>
		switch(c) {
    16c8:	7101901f 	cmp	w0, #0x64
    16cc:	54000dc1 	b.ne	1884 <main+0x24c>  // b.any
				sscanf(optarg, "%d", &dev_id);
    16d0:	910253e2 	add	x2, sp, #0x94
    16d4:	f0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    16d8:	912ae021 	add	x1, x1, #0xab8
    16dc:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    16e0:	f947dc00 	ldr	x0, [x0, #4024]
    16e4:	f9400000 	ldr	x0, [x0]
    16e8:	97ffff2e 	bl	13a0 <__isoc99_sscanf@plt>
				if(dev_id < -2 || dev_id >= 0) {
    16ec:	b94097e0 	ldr	w0, [sp, #148]
    16f0:	11000800 	add	w0, w0, #0x2
    16f4:	7100041f 	cmp	w0, #0x1
    16f8:	540005a9 	b.ls	17ac <main+0x174>  // b.plast
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
    16fc:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    1700:	f947d800 	ldr	x0, [x0, #4016]
    1704:	f9400003 	ldr	x3, [x0]
    1708:	d2800382 	mov	x2, #0x1c                  	// #28
    170c:	d2800021 	mov	x1, #0x1                   	// #1
    1710:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1714:	912b0000 	add	x0, x0, #0xac0
    1718:	97ffff12 	bl	1360 <fwrite@plt>
					exit(1);
    171c:	52800020 	mov	w0, #0x1                   	// #1
    1720:	97fffe98 	bl	1180 <exit@plt>
		switch(c) {
    1724:	7101c81f 	cmp	w0, #0x72
    1728:	540009e0 	b.eq	1864 <main+0x22c>  // b.none
    172c:	7101d81f 	cmp	w0, #0x76
    1730:	54000aa1 	b.ne	1884 <main+0x24c>  // b.any
				strcpy(fvname, optarg);
    1734:	911323f5 	add	x21, sp, #0x4c8
    1738:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    173c:	f947dc00 	ldr	x0, [x0, #4024]
    1740:	f9400001 	ldr	x1, [x0]
    1744:	aa1503e0 	mov	x0, x21
    1748:	97ffff0e 	bl	1380 <strcpy@plt>
				printf("validation input file: %s\n", fvname); fflush(stdout);
    174c:	aa1503e1 	mov	x1, x21
    1750:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1754:	912ba000 	add	x0, x0, #0xae8
    1758:	97ffff16 	bl	13b0 <printf@plt>
    175c:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    1760:	f947e000 	ldr	x0, [x0, #4032]
    1764:	f9400000 	ldr	x0, [x0]
    1768:	97ffff02 	bl	1370 <fflush@plt>
				random = false;
    176c:	52800015 	mov	w21, #0x0                   	// #0
	for(;;) {
    1770:	1400000f 	b	17ac <main+0x174>
				strcpy(fname, optarg);
    1774:	9122c3fb 	add	x27, sp, #0x8b0
    1778:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    177c:	f947dc00 	ldr	x0, [x0, #4024]
    1780:	f9400001 	ldr	x1, [x0]
    1784:	aa1b03e0 	mov	x0, x27
    1788:	97fffefe 	bl	1380 <strcpy@plt>
				printf("input file: %s\n", fname); fflush(stdout);
    178c:	aa1b03e1 	mov	x1, x27
    1790:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1794:	912a2000 	add	x0, x0, #0xa88
    1798:	97ffff06 	bl	13b0 <printf@plt>
    179c:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    17a0:	f947e000 	ldr	x0, [x0, #4032]
    17a4:	f9400000 	ldr	x0, [x0]
    17a8:	97fffef2 	bl	1370 <fflush@plt>
		int option_index = 0;
    17ac:	b90087ff 	str	wzr, [sp, #132]
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    17b0:	aa1903e4 	mov	x4, x25
    17b4:	aa1803e3 	mov	x3, x24
    17b8:	aa1703e2 	mov	x2, x23
    17bc:	aa1403e1 	mov	x1, x20
    17c0:	2a1303e0 	mov	w0, w19
    17c4:	97fffed7 	bl	1320 <getopt_long@plt>
		if(c == -1) {
    17c8:	3100041f 	cmn	w0, #0x1
    17cc:	54000640 	b.eq	1894 <main+0x25c>  // b.none
		switch(c) {
    17d0:	7101bc1f 	cmp	w0, #0x6f
    17d4:	540001c0 	b.eq	180c <main+0x1d4>  // b.none
    17d8:	54fffa6c 	b.gt	1724 <main+0xec>
    17dc:	7101a41f 	cmp	w0, #0x69
    17e0:	54fffca0 	b.eq	1774 <main+0x13c>  // b.none
    17e4:	7101b41f 	cmp	w0, #0x6d
    17e8:	54fff701 	b.ne	16c8 <main+0x90>  // b.any
				sscanf(optarg, "%"PASTA_SCN_INDEX, &mode);
    17ec:	910273e2 	add	x2, sp, #0x9c
    17f0:	f0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    17f4:	912ac021 	add	x1, x1, #0xab0
    17f8:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    17fc:	f947dc00 	ldr	x0, [x0, #4024]
    1800:	f9400000 	ldr	x0, [x0]
    1804:	97fffee7 	bl	13a0 <__isoc99_sscanf@plt>
				break;
    1808:	17ffffe9 	b	17ac <main+0x174>
				fo = fopen(optarg, "w");
    180c:	b00000bb 	adrp	x27, 16000 <__FRAME_END__+0x100e8>
    1810:	f947df7b 	ldr	x27, [x27, #4024]
    1814:	aa1a03e1 	mov	x1, x26
    1818:	f9400360 	ldr	x0, [x27]
    181c:	97fffe79 	bl	1200 <fopen@plt>
    1820:	aa0003f6 	mov	x22, x0
				strcpy(foname, optarg);
    1824:	f940037b 	ldr	x27, [x27]
    1828:	aa1b03e1 	mov	x1, x27
    182c:	910383e0 	add	x0, sp, #0xe0
    1830:	97fffed4 	bl	1380 <strcpy@plt>
				sptAssert(fo != NULL);
    1834:	b4000156 	cbz	x22, 185c <main+0x224>
				printf("output file: %s\n", optarg); fflush(stdout);
    1838:	aa1b03e1 	mov	x1, x27
    183c:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1840:	912a6000 	add	x0, x0, #0xa98
    1844:	97fffedb 	bl	13b0 <printf@plt>
    1848:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    184c:	f947e000 	ldr	x0, [x0, #4032]
    1850:	f9400000 	ldr	x0, [x0]
    1854:	97fffec7 	bl	1370 <fflush@plt>
				break;
    1858:	17ffffd5 	b	17ac <main+0x174>
				sptAssert(fo != NULL);
    185c:	12800000 	mov	w0, #0xffffffff            	// #-1
    1860:	97fffe48 	bl	1180 <exit@plt>
				sscanf(optarg, "%u"PASTA_SCN_INDEX, &R);
    1864:	910263e2 	add	x2, sp, #0x98
    1868:	f0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    186c:	912b8021 	add	x1, x1, #0xae0
    1870:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    1874:	f947dc00 	ldr	x0, [x0, #4024]
    1878:	f9400000 	ldr	x0, [x0]
    187c:	97fffec9 	bl	13a0 <__isoc99_sscanf@plt>
				break;
    1880:	17ffffcb 	b	17ac <main+0x174>
				print_usage(argv);
    1884:	aa1403e0 	mov	x0, x20
    1888:	97ffff34 	bl	1558 <print_usage>
				exit(1);
    188c:	52800020 	mov	w0, #0x1                   	// #1
    1890:	97fffe3c 	bl	1180 <exit@plt>
	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1894:	b9409fe1 	ldr	w1, [sp, #156]
    1898:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    189c:	912c2000 	add	x0, x0, #0xb08
    18a0:	97fffec4 	bl	13b0 <printf@plt>
	printf("dev_id: %d\n", dev_id);
    18a4:	b94097e1 	ldr	w1, [sp, #148]
    18a8:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    18ac:	912c6000 	add	x0, x0, #0xb18
    18b0:	97fffec0 	bl	13b0 <printf@plt>
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    18b4:	9122c3e2 	add	x2, sp, #0x8b0
    18b8:	52800021 	mov	w1, #0x1                   	// #1
    18bc:	910283e0 	add	x0, sp, #0xa0
    18c0:	94000394 	bl	2710 <sptLoadSparseTensor>
    18c4:	35000ae0 	cbnz	w0, 1a20 <main+0x3e8>
	sptSparseTensorStatus(&X, stdout);
    18c8:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    18cc:	f947e000 	ldr	x0, [x0, #4032]
    18d0:	f9400001 	ldr	x1, [x0]
    18d4:	910283e0 	add	x0, sp, #0xa0
    18d8:	94000642 	bl	31e0 <sptSparseTensorStatus>
	sptIndex nmodes = X.nmodes;
    18dc:	b940a3f3 	ldr	w19, [sp, #160]
	U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    18e0:	11000677 	add	w23, w19, #0x1
    18e4:	11000660 	add	w0, w19, #0x1
    18e8:	d37df000 	lsl	x0, x0, #3
    18ec:	97fffe4d 	bl	1220 <malloc@plt>
    18f0:	aa0003f4 	mov	x20, x0
	for(sptIndex m=0; m<nmodes+1; ++m) {
    18f4:	34000197 	cbz	w23, 1924 <main+0x2ec>
    18f8:	aa0003f7 	mov	x23, x0
    18fc:	d37d7e78 	ubfiz	x24, x19, #3, #32
    1900:	91002318 	add	x24, x24, #0x8
    1904:	8b000318 	add	x24, x24, x0
		U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    1908:	d2800319 	mov	x25, #0x18                  	// #24
    190c:	aa1903e0 	mov	x0, x25
    1910:	97fffe44 	bl	1220 <malloc@plt>
    1914:	f80086e0 	str	x0, [x23], #8
	for(sptIndex m=0; m<nmodes+1; ++m) {
    1918:	eb1802ff 	cmp	x23, x24
    191c:	54ffff81 	b.ne	190c <main+0x2d4>  // b.any
	for(sptIndex m=0; m<nmodes; ++m) {
    1920:	340008d3 	cbz	w19, 1a38 <main+0x400>
				random = false;
    1924:	d2800017 	mov	x23, #0x0                   	// #0
    1928:	52800018 	mov	w24, #0x0                   	// #0
		sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
    192c:	d37ef6f9 	lsl	x25, x23, #2
    1930:	b9409be2 	ldr	w2, [sp, #152]
    1934:	f9405be0 	ldr	x0, [sp, #176]
    1938:	b8777801 	ldr	w1, [x0, x23, lsl #2]
    193c:	f8777a80 	ldr	x0, [x20, x23, lsl #3]
    1940:	9400057d 	bl	2f34 <sptNewMatrix>
    1944:	35000720 	cbnz	w0, 1a28 <main+0x3f0>
		sptAssert(sptRandomizeMatrix(U[m], random) == 0);
    1948:	2a1503e1 	mov	w1, w21
    194c:	f8777a80 	ldr	x0, [x20, x23, lsl #3]
    1950:	940005ab 	bl	2ffc <sptRandomizeMatrix>
    1954:	350006e0 	cbnz	w0, 1a30 <main+0x3f8>
		if(X.ndims[m] > max_ndims)
    1958:	f9405be0 	ldr	x0, [sp, #176]
    195c:	b8796800 	ldr	w0, [x0, x25]
    1960:	6b00031f 	cmp	w24, w0
    1964:	1a802318 	csel	w24, w24, w0, cs  // cs = hs, nlast
	for(sptIndex m=0; m<nmodes; ++m) {
    1968:	910006f7 	add	x23, x23, #0x1
    196c:	6b17027f 	cmp	w19, w23
    1970:	54fffde8 	b.hi	192c <main+0x2f4>  // b.pmore
	sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    1974:	2a1303fa 	mov	w26, w19
    1978:	d37d7e7c 	ubfiz	x28, x19, #3, #32
    197c:	8b1c0299 	add	x25, x20, x28
    1980:	b9409be2 	ldr	w2, [sp, #152]
    1984:	2a1803e1 	mov	w1, w24
    1988:	f87a7a80 	ldr	x0, [x20, x26, lsl #3]
    198c:	9400056a 	bl	2f34 <sptNewMatrix>
    1990:	35000580 	cbnz	w0, 1a40 <main+0x408>
	sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1994:	0f000400 	movi	v0.2s, #0x0
    1998:	f87a7a80 	ldr	x0, [x20, x26, lsl #3]
    199c:	940005d2 	bl	30e4 <sptConstantMatrix>
    19a0:	35000540 	cbnz	w0, 1a48 <main+0x410>
	sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
    19a4:	d37ef740 	lsl	x0, x26, #2
    19a8:	97fffe1e 	bl	1220 <malloc@plt>
    19ac:	aa0003f8 	mov	x24, x0
	mats_order[0] = mode;
    19b0:	b9409fe3 	ldr	w3, [sp, #156]
    19b4:	b9000003 	str	w3, [x0]
	for(sptIndex i=1; i<nmodes; ++i)
    19b8:	7100067f 	cmp	w19, #0x1
    19bc:	54000129 	b.ls	19e0 <main+0x3a8>  // b.plast
    19c0:	d2800020 	mov	x0, #0x1                   	// #1
		mats_order[i] = (mode+i) % nmodes;
    19c4:	0b000062 	add	w2, w3, w0
    19c8:	1ad30841 	udiv	w1, w2, w19
    19cc:	1b138821 	msub	w1, w1, w19, w2
    19d0:	b8207b01 	str	w1, [x24, x0, lsl #2]
	for(sptIndex i=1; i<nmodes; ++i)
    19d4:	91000400 	add	x0, x0, #0x1
    19d8:	6b00027f 	cmp	w19, w0
    19dc:	54ffff48 	b.hi	19c4 <main+0x38c>  // b.pmore
	if(dev_id == -2) {
    19e0:	b94097e0 	ldr	w0, [sp, #148]
    19e4:	3100081f 	cmn	w0, #0x2
    19e8:	54000340 	b.eq	1a50 <main+0x418>  // b.none
	int nthreads = 1;
    19ec:	52800021 	mov	w1, #0x1                   	// #1
    19f0:	b9007fe1 	str	w1, [sp, #124]
	} else if(dev_id == -1) {
    19f4:	3100041f 	cmn	w0, #0x1
    19f8:	540003e0 	b.eq	1a74 <main+0x43c>  // b.none
	sptNewTimer(&timer, 0);
    19fc:	52800001 	mov	w1, #0x0                   	// #0
    1a00:	910223e0 	add	x0, sp, #0x88
    1a04:	940009ae 	bl	40bc <sptNewTimer>
	sptStartTimer(timer);
    1a08:	f94047e0 	ldr	x0, [sp, #136]
    1a0c:	940009c6 	bl	4124 <sptStartTimer>
    1a10:	528000b7 	mov	w23, #0x5                   	// #5
		sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1a14:	0f000408 	movi	v8.2s, #0x0
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1a18:	910283fb 	add	x27, sp, #0xa0
    1a1c:	14000037 	b	1af8 <main+0x4c0>
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1a20:	12800000 	mov	w0, #0xffffffff            	// #-1
    1a24:	97fffdd7 	bl	1180 <exit@plt>
		sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
    1a28:	12800000 	mov	w0, #0xffffffff            	// #-1
    1a2c:	97fffdd5 	bl	1180 <exit@plt>
		sptAssert(sptRandomizeMatrix(U[m], random) == 0);
    1a30:	12800000 	mov	w0, #0xffffffff            	// #-1
    1a34:	97fffdd3 	bl	1180 <exit@plt>
	sptIndex max_ndims = 0;
    1a38:	2a1303f8 	mov	w24, w19
    1a3c:	17ffffce 	b	1974 <main+0x33c>
	sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    1a40:	12800000 	mov	w0, #0xffffffff            	// #-1
    1a44:	97fffdcf 	bl	1180 <exit@plt>
	sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1a48:	12800000 	mov	w0, #0xffffffff            	// #-1
    1a4c:	97fffdcd 	bl	1180 <exit@plt>
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1a50:	aa1803e2 	mov	x2, x24
    1a54:	aa1403e1 	mov	x1, x20
    1a58:	910283e0 	add	x0, sp, #0xa0
    1a5c:	940006db 	bl	35c8 <sptMTTKRP>
		nthreads = 1;
    1a60:	52800021 	mov	w1, #0x1                   	// #1
    1a64:	b9007fe1 	str	w1, [sp, #124]
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1a68:	34fffca0 	cbz	w0, 19fc <main+0x3c4>
    1a6c:	12800000 	mov	w0, #0xffffffff            	// #-1
    1a70:	97fffdc4 	bl	1180 <exit@plt>
		#pragma omp parallel
    1a74:	b90c9be1 	str	w1, [sp, #3224]
    1a78:	52800003 	mov	w3, #0x0                   	// #0
    1a7c:	52800002 	mov	w2, #0x0                   	// #0
    1a80:	913263e1 	add	x1, sp, #0xc98
    1a84:	90000000 	adrp	x0, 1000 <g_last_error+0x1000>
    1a88:	9114d000 	add	x0, x0, #0x534
    1a8c:	97fffe51 	bl	13d0 <GOMP_parallel@plt>
    1a90:	b94c9be0 	ldr	w0, [sp, #3224]
    1a94:	2a0003f7 	mov	w23, w0
    1a98:	b9007fe0 	str	w0, [sp, #124]
        printf("\nnthreads: %d\n", nthreads);
    1a9c:	2a0003e1 	mov	w1, w0
    1aa0:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1aa4:	912ca000 	add	x0, x0, #0xb28
    1aa8:	97fffe42 	bl	13b0 <printf@plt>
        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1aac:	2a1703e4 	mov	w4, w23
    1ab0:	b9409fe3 	ldr	w3, [sp, #156]
    1ab4:	aa1803e2 	mov	x2, x24
    1ab8:	aa1403e1 	mov	x1, x20
    1abc:	910283e0 	add	x0, sp, #0xa0
    1ac0:	94000901 	bl	3ec4 <sptOmpMTTKRP>
    1ac4:	34fff9c0 	cbz	w0, 19fc <main+0x3c4>
    1ac8:	12800000 	mov	w0, #0xffffffff            	// #-1
    1acc:	97fffdad 	bl	1180 <exit@plt>
		sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1ad0:	12800000 	mov	w0, #0xffffffff            	// #-1
    1ad4:	97fffdab 	bl	1180 <exit@plt>
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1ad8:	b9409fe3 	ldr	w3, [sp, #156]
    1adc:	aa1803e2 	mov	x2, x24
    1ae0:	aa1403e1 	mov	x1, x20
    1ae4:	aa1b03e0 	mov	x0, x27
    1ae8:	940006b8 	bl	35c8 <sptMTTKRP>
    1aec:	350002a0 	cbnz	w0, 1b40 <main+0x508>
	for(int it=0; it<niters; ++it) {
    1af0:	710006f7 	subs	w23, w23, #0x1
    1af4:	540002a0 	b.eq	1b48 <main+0x510>  // b.none
		sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1af8:	1e204100 	fmov	s0, s8
    1afc:	f9400320 	ldr	x0, [x25]
    1b00:	94000579 	bl	30e4 <sptConstantMatrix>
    1b04:	35fffe60 	cbnz	w0, 1ad0 <main+0x498>
		if(dev_id == -2) {
    1b08:	b94097e0 	ldr	w0, [sp, #148]
    1b0c:	3100081f 	cmn	w0, #0x2
    1b10:	54fffe40 	b.eq	1ad8 <main+0x4a0>  // b.none
		} else if(dev_id == -1) {
    1b14:	3100041f 	cmn	w0, #0x1
    1b18:	54fffec1 	b.ne	1af0 <main+0x4b8>  // b.any
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1b1c:	b9407fe4 	ldr	w4, [sp, #124]
    1b20:	b9409fe3 	ldr	w3, [sp, #156]
    1b24:	aa1803e2 	mov	x2, x24
    1b28:	aa1403e1 	mov	x1, x20
    1b2c:	aa1b03e0 	mov	x0, x27
    1b30:	940008e5 	bl	3ec4 <sptOmpMTTKRP>
    1b34:	34fffde0 	cbz	w0, 1af0 <main+0x4b8>
    1b38:	12800000 	mov	w0, #0xffffffff            	// #-1
    1b3c:	97fffd91 	bl	1180 <exit@plt>
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1b40:	12800000 	mov	w0, #0xffffffff            	// #-1
    1b44:	97fffd8f 	bl	1180 <exit@plt>
	sptStopTimer(timer);
    1b48:	f94047e0 	ldr	x0, [sp, #136]
    1b4c:	940009a6 	bl	41e4 <sptStopTimer>
	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
    1b50:	f0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    1b54:	912ce042 	add	x2, x2, #0xb38
    1b58:	528000a1 	mov	w1, #0x5                   	// #5
    1b5c:	f94047e0 	ldr	x0, [sp, #136]
    1b60:	940009f8 	bl	4340 <sptPrintAverageElapsedTime>
    1b64:	1e604003 	fmov	d3, d0
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1b68:	b9409be3 	ldr	w3, [sp, #152]
    1b6c:	f9405fe2 	ldr	x2, [sp, #184]
    1b70:	1e630262 	ucvtf	d2, w19
    1b74:	1e630060 	ucvtf	d0, w3
    1b78:	1e600842 	fmul	d2, d2, d0
    1b7c:	9e630040 	ucvtf	d0, x2
    1b80:	1e600842 	fmul	d2, d2, d0
    1b84:	1e631842 	fdiv	d2, d2, d3
    1b88:	d2d9aca0 	mov	x0, #0xcd6500000000        	// #225833675390976
    1b8c:	f2e839a0 	movk	x0, #0x41cd, lsl #48
    1b90:	9e670000 	fmov	d0, x0
    1b94:	1e601840 	fdiv	d0, d2, d0
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    1b98:	9b020b42 	madd	x2, x26, x2, x2
    1b9c:	d37ef442 	lsl	x2, x2, #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1ba0:	34000c93 	cbz	w19, 1d30 <main+0x6f8>
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1ba4:	f9405be4 	ldr	x4, [sp, #176]
    1ba8:	d2800000 	mov	x0, #0x0                   	// #0
    1bac:	b8607881 	ldr	w1, [x4, x0, lsl #2]
    1bb0:	1b017c61 	mul	w1, w3, w1
    1bb4:	8b010842 	add	x2, x2, x1, lsl #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1bb8:	91000400 	add	x0, x0, #0x1
    1bbc:	6b00027f 	cmp	w19, w0
    1bc0:	54ffff68 	b.hi	1bac <main+0x574>  // b.pmore
	double gbw = (double)bytes / aver_time / 1e9;
    1bc4:	9e630041 	ucvtf	d1, x2
    1bc8:	1e631821 	fdiv	d1, d1, d3
    1bcc:	d2d9aca0 	mov	x0, #0xcd6500000000        	// #225833675390976
    1bd0:	f2e839a0 	movk	x0, #0x41cd, lsl #48
    1bd4:	9e670002 	fmov	d2, x0
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1bd8:	1e621821 	fdiv	d1, d1, d2
    1bdc:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1be0:	91302000 	add	x0, x0, #0xc08
    1be4:	97fffdf3 	bl	13b0 <printf@plt>
	if(fo != NULL) {
    1be8:	b4000996 	cbz	x22, 1d18 <main+0x6e0>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1bec:	aa1603e1 	mov	x1, x22
    1bf0:	f87c6a80 	ldr	x0, [x20, x28]
    1bf4:	940009ff 	bl	43f0 <sptDumpMatrix>
    1bf8:	350003c0 	cbnz	w0, 1c70 <main+0x638>
		fclose(fo);
    1bfc:	aa1603e0 	mov	x0, x22
    1c00:	97fffd7c 	bl	11f0 <fclose@plt>
	sptFreeTimer(timer);
    1c04:	f94047e0 	ldr	x0, [sp, #136]
    1c08:	940009e4 	bl	4398 <sptFreeTimer>
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1c0c:	d2800016 	mov	x22, #0x0                   	// #0
		sptFreeMatrix(U[m]);
    1c10:	f8767a80 	ldr	x0, [x20, x22, lsl #3]
    1c14:	94000549 	bl	3138 <sptFreeMatrix>
	for(sptIndex m=0; m<nmodes; ++m) {
    1c18:	910006d6 	add	x22, x22, #0x1
    1c1c:	6b16027f 	cmp	w19, w22
    1c20:	54ffff88 	b.hi	1c10 <main+0x5d8>  // b.pmore
	sptFreeSparseTensor(&X);
    1c24:	910283e0 	add	x0, sp, #0xa0
    1c28:	940000ab 	bl	1ed4 <sptFreeSparseTensor>
	free(mats_order);
    1c2c:	aa1803e0 	mov	x0, x24
    1c30:	97fffdc8 	bl	1350 <free@plt>
	sptFreeMatrix(U[nmodes]);
    1c34:	f87a7a80 	ldr	x0, [x20, x26, lsl #3]
    1c38:	94000540 	bl	3138 <sptFreeMatrix>
	free(U);
    1c3c:	aa1403e0 	mov	x0, x20
    1c40:	97fffdc4 	bl	1350 <free@plt>
	if (!random){
    1c44:	340001b5 	cbz	w21, 1c78 <main+0x640>
}
    1c48:	52800000 	mov	w0, #0x0                   	// #0
    1c4c:	fd4033e8 	ldr	d8, [sp, #96]
    1c50:	a94153f3 	ldp	x19, x20, [sp, #16]
    1c54:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1c58:	a94363f7 	ldp	x23, x24, [sp, #48]
    1c5c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    1c60:	a94573fb 	ldp	x27, x28, [sp, #80]
    1c64:	a9407bfd 	ldp	x29, x30, [sp]
    1c68:	913283ff 	add	sp, sp, #0xca0
    1c6c:	d65f03c0 	ret
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1c70:	12800000 	mov	w0, #0xffffffff            	// #-1
    1c74:	97fffd43 	bl	1180 <exit@plt>
		FILE* fPtr1 = fopen(fvname, "r");
    1c78:	f0000013 	adrp	x19, 4000 <sptOmpMTTKRP+0x13c>
    1c7c:	912d4273 	add	x19, x19, #0xb50
    1c80:	aa1303e1 	mov	x1, x19
    1c84:	911323e0 	add	x0, sp, #0x4c8
    1c88:	97fffd5e 	bl	1200 <fopen@plt>
    1c8c:	aa0003f4 	mov	x20, x0
		FILE* fPtr2 = fopen(foname, "r");
    1c90:	aa1303e1 	mov	x1, x19
    1c94:	910383e0 	add	x0, sp, #0xe0
    1c98:	97fffd5a 	bl	1200 <fopen@plt>
    1c9c:	aa0003f3 	mov	x19, x0
		if (fPtr1 == NULL || fPtr2 == NULL) {
    1ca0:	f100029f 	cmp	x20, #0x0
    1ca4:	fa401804 	ccmp	x0, #0x0, #0x4, ne  // ne = any
    1ca8:	54000121 	b.ne	1ccc <main+0x694>  // b.any
			printf("\nUnable to open file.\n");
    1cac:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1cb0:	912d6000 	add	x0, x0, #0xb58
    1cb4:	97fffd97 	bl	1310 <puts@plt>
			printf("Please check whether file exists and you have read privilege.\n");
    1cb8:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1cbc:	912dc000 	add	x0, x0, #0xb70
    1cc0:	97fffd94 	bl	1310 <puts@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    1cc4:	b40001b4 	cbz	x20, 1cf8 <main+0x6c0>
    1cc8:	1400000a 	b	1cf0 <main+0x6b8>
			int diff = compareFile(fPtr1, fPtr2);
    1ccc:	aa0003e1 	mov	x1, x0
    1cd0:	aa1403e0 	mov	x0, x20
    1cd4:	97fffe40 	bl	15d4 <compareFile>
			if (diff == 0) {
    1cd8:	35000180 	cbnz	w0, 1d08 <main+0x6d0>
				printf("Validation Successful \n %s matchs %s\n", foname, fvname);
    1cdc:	911323e2 	add	x2, sp, #0x4c8
    1ce0:	910383e1 	add	x1, sp, #0xe0
    1ce4:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1ce8:	912ec000 	add	x0, x0, #0xbb0
    1cec:	97fffdb1 	bl	13b0 <printf@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    1cf0:	aa1403e0 	mov	x0, x20
    1cf4:	97fffd3f 	bl	11f0 <fclose@plt>
		if(fPtr2 != NULL) fclose(fPtr2);
    1cf8:	b4fffa93 	cbz	x19, 1c48 <main+0x610>
    1cfc:	aa1303e0 	mov	x0, x19
    1d00:	97fffd3c 	bl	11f0 <fclose@plt>
	return 0;
    1d04:	17ffffd1 	b	1c48 <main+0x610>
				printf("\nFiles are not equal.\n Validation FAILED \n");
    1d08:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1d0c:	912f6000 	add	x0, x0, #0xbd8
    1d10:	97fffd80 	bl	1310 <puts@plt>
    1d14:	17fffff7 	b	1cf0 <main+0x6b8>
	sptFreeTimer(timer);
    1d18:	f94047e0 	ldr	x0, [sp, #136]
    1d1c:	9400099f 	bl	4398 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1d20:	17ffffbb 	b	1c0c <main+0x5d4>
	sptFreeTimer(timer);
    1d24:	f94047e0 	ldr	x0, [sp, #136]
    1d28:	9400099c 	bl	4398 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1d2c:	17ffffbe 	b	1c24 <main+0x5ec>
	double gbw = (double)bytes / aver_time / 1e9;
    1d30:	9e630042 	ucvtf	d2, x2
    1d34:	1e631841 	fdiv	d1, d2, d3
    1d38:	d2d9aca0 	mov	x0, #0xcd6500000000        	// #225833675390976
    1d3c:	f2e839a0 	movk	x0, #0x41cd, lsl #48
    1d40:	9e670002 	fmov	d2, x0
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1d44:	1e621821 	fdiv	d1, d1, d2
    1d48:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1d4c:	91302000 	add	x0, x0, #0xc08
    1d50:	97fffd98 	bl	13b0 <printf@plt>
	if(fo != NULL) {
    1d54:	b4fffe96 	cbz	x22, 1d24 <main+0x6ec>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1d58:	aa1603e1 	mov	x1, x22
    1d5c:	f87c6a80 	ldr	x0, [x20, x28]
    1d60:	940009a4 	bl	43f0 <sptDumpMatrix>
    1d64:	35fff860 	cbnz	w0, 1c70 <main+0x638>
		fclose(fo);
    1d68:	aa1603e0 	mov	x0, x22
    1d6c:	97fffd21 	bl	11f0 <fclose@plt>
	sptFreeTimer(timer);
    1d70:	f94047e0 	ldr	x0, [sp, #136]
    1d74:	94000989 	bl	4398 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1d78:	17ffffab 	b	1c24 <main+0x5ec>

0000000000001d7c <sptNewSparseTensor>:
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1d7c:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    1d80:	910003fd 	mov	x29, sp
    1d84:	a90153f3 	stp	x19, x20, [sp, #16]
    1d88:	a9025bf5 	stp	x21, x22, [sp, #32]
    1d8c:	f9001bf7 	str	x23, [sp, #48]
    1d90:	aa0003f6 	mov	x22, x0
    1d94:	2a0103f3 	mov	w19, w1
    1d98:	aa0203f5 	mov	x21, x2
	sptIndex i;
	int result;
	tsr->nmodes = nmodes;
    1d9c:	b9000001 	str	w1, [x0]
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1da0:	2a0103f4 	mov	w20, w1
    1da4:	d37e7e77 	ubfiz	x23, x19, #2, #32
    1da8:	aa1703e0 	mov	x0, x23
    1dac:	97fffd1d 	bl	1220 <malloc@plt>
    1db0:	f90006c0 	str	x0, [x22, #8]
	for(i = 0; i < nmodes; ++i) {
    1db4:	340007b3 	cbz	w19, 1ea8 <sptNewSparseTensor+0x12c>
    1db8:	d2800003 	mov	x3, #0x0                   	// #0
		tsr->sortorder[i] = i;
    1dbc:	f94006c0 	ldr	x0, [x22, #8]
    1dc0:	b8237803 	str	w3, [x0, x3, lsl #2]
	for(i = 0; i < nmodes; ++i) {
    1dc4:	91000463 	add	x3, x3, #0x1
    1dc8:	6b03027f 	cmp	w19, w3
    1dcc:	54ffff88 	b.hi	1dbc <sptNewSparseTensor+0x40>  // b.pmore
	}
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1dd0:	aa1703e0 	mov	x0, x23
    1dd4:	97fffd13 	bl	1220 <malloc@plt>
    1dd8:	f9000ac0 	str	x0, [x22, #16]
//	spt_CheckOSError(!tsr->ndims, "SpTns New");
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1ddc:	aa1703e2 	mov	x2, x23
    1de0:	aa1503e1 	mov	x1, x21
    1de4:	97fffce3 	bl	1170 <memcpy@plt>
	tsr->nnz = 0;
    1de8:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1dec:	8b140694 	add	x20, x20, x20, lsl #1
    1df0:	d37df280 	lsl	x0, x20, #3
    1df4:	97fffd0b 	bl	1220 <malloc@plt>
    1df8:	f90012c0 	str	x0, [x22, #32]
//	spt_CheckOSError(!tsr->inds, "SpTns New");
	for(i = 0; i < nmodes; ++i) {
    1dfc:	d37f7e75 	ubfiz	x21, x19, #1, #32
    1e00:	8b3342b3 	add	x19, x21, w19, uxtw
    1e04:	d37df275 	lsl	x21, x19, #3
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1e08:	d2800014 	mov	x20, #0x0                   	// #0
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1e0c:	f94012c0 	ldr	x0, [x22, #32]
    1e10:	d2800002 	mov	x2, #0x0                   	// #0
    1e14:	d2800001 	mov	x1, #0x0                   	// #0
    1e18:	8b140000 	add	x0, x0, x20
    1e1c:	940000f3 	bl	21e8 <sptNewIndexVector>
    1e20:	2a0003f3 	mov	w19, w0
		spt_CheckError(result, "SpTns New", NULL);
    1e24:	35000260 	cbnz	w0, 1e70 <sptNewSparseTensor+0xf4>
	for(i = 0; i < nmodes; ++i) {
    1e28:	91006294 	add	x20, x20, #0x18
    1e2c:	eb1402bf 	cmp	x21, x20
    1e30:	54fffee1 	b.ne	1e0c <sptNewSparseTensor+0x90>  // b.any
	}
	result = sptNewValueVector(&tsr->values, 0, 0);
    1e34:	d2800002 	mov	x2, #0x0                   	// #0
    1e38:	d2800001 	mov	x1, #0x0                   	// #0
    1e3c:	9100a2c0 	add	x0, x22, #0x28
    1e40:	94000041 	bl	1f44 <sptNewValueVector>
    1e44:	2a0003f3 	mov	w19, w0
	spt_CheckError(result, "SpTns New", NULL);
    1e48:	34000240 	cbz	w0, 1e90 <sptNewSparseTensor+0x114>
    1e4c:	d2800004 	mov	x4, #0x0                   	// #0
    1e50:	528006a3 	mov	w3, #0x35                  	// #53
    1e54:	f0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    1e58:	91324042 	add	x2, x2, #0xc90
    1e5c:	2a0003e1 	mov	w1, w0
    1e60:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1e64:	9132e000 	add	x0, x0, #0xcb8
    1e68:	940009c1 	bl	456c <spt_ComplainError>
    1e6c:	14000009 	b	1e90 <sptNewSparseTensor+0x114>
		spt_CheckError(result, "SpTns New", NULL);
    1e70:	d2800004 	mov	x4, #0x0                   	// #0
    1e74:	52800643 	mov	w3, #0x32                  	// #50
    1e78:	f0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    1e7c:	91324042 	add	x2, x2, #0xc90
    1e80:	2a0003e1 	mov	w1, w0
    1e84:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1e88:	9132e000 	add	x0, x0, #0xcb8
    1e8c:	940009b8 	bl	456c <spt_ComplainError>
	return 0;
}
    1e90:	2a1303e0 	mov	w0, w19
    1e94:	a94153f3 	ldp	x19, x20, [sp, #16]
    1e98:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1e9c:	f9401bf7 	ldr	x23, [sp, #48]
    1ea0:	a8c47bfd 	ldp	x29, x30, [sp], #64
    1ea4:	d65f03c0 	ret
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1ea8:	aa1703e0 	mov	x0, x23
    1eac:	97fffcdd 	bl	1220 <malloc@plt>
    1eb0:	f9000ac0 	str	x0, [x22, #16]
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1eb4:	aa1703e2 	mov	x2, x23
    1eb8:	aa1503e1 	mov	x1, x21
    1ebc:	97fffcad 	bl	1170 <memcpy@plt>
	tsr->nnz = 0;
    1ec0:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1ec4:	d2800000 	mov	x0, #0x0                   	// #0
    1ec8:	97fffcd6 	bl	1220 <malloc@plt>
    1ecc:	f90012c0 	str	x0, [x22, #32]
	for(i = 0; i < nmodes; ++i) {
    1ed0:	17ffffd9 	b	1e34 <sptNewSparseTensor+0xb8>

0000000000001ed4 <sptFreeSparseTensor>:

/**
 * Release any memory the sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSparseTensor(sptSparseTensor *tsr) {
    1ed4:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1ed8:	910003fd 	mov	x29, sp
    1edc:	a90153f3 	stp	x19, x20, [sp, #16]
    1ee0:	aa0003f4 	mov	x20, x0
	sptIndex i;
	for(i = 0; i < tsr->nmodes; ++i) {
    1ee4:	b9400000 	ldr	w0, [x0]
    1ee8:	34000160 	cbz	w0, 1f14 <sptFreeSparseTensor+0x40>
    1eec:	52800013 	mov	w19, #0x0                   	// #0
		sptFreeIndexVector(&tsr->inds[i]);
    1ef0:	d37f7e61 	ubfiz	x1, x19, #1, #32
    1ef4:	8b334021 	add	x1, x1, w19, uxtw
    1ef8:	f9401280 	ldr	x0, [x20, #32]
    1efc:	8b010c00 	add	x0, x0, x1, lsl #3
    1f00:	9400014d 	bl	2434 <sptFreeIndexVector>
	for(i = 0; i < tsr->nmodes; ++i) {
    1f04:	11000673 	add	w19, w19, #0x1
    1f08:	b9400280 	ldr	w0, [x20]
    1f0c:	6b13001f 	cmp	w0, w19
    1f10:	54ffff08 	b.hi	1ef0 <sptFreeSparseTensor+0x1c>  // b.pmore
	}
	free(tsr->sortorder);
    1f14:	f9400680 	ldr	x0, [x20, #8]
    1f18:	97fffd0e 	bl	1350 <free@plt>
	free(tsr->ndims);
    1f1c:	f9400a80 	ldr	x0, [x20, #16]
    1f20:	97fffd0c 	bl	1350 <free@plt>
	free(tsr->inds);
    1f24:	f9401280 	ldr	x0, [x20, #32]
    1f28:	97fffd0a 	bl	1350 <free@plt>
	sptFreeValueVector(&tsr->values);
    1f2c:	9100a280 	add	x0, x20, #0x28
    1f30:	940000a6 	bl	21c8 <sptFreeValueVector>
	tsr->nmodes = 0;
    1f34:	b900029f 	str	wzr, [x20]
}
    1f38:	a94153f3 	ldp	x19, x20, [sp, #16]
    1f3c:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1f40:	d65f03c0 	ret

0000000000001f44 <sptNewValueVector>:
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptNewValueVector(sptValueVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    1f44:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1f48:	910003fd 	mov	x29, sp
    1f4c:	a90153f3 	stp	x19, x20, [sp, #16]
    1f50:	a9025bf5 	stp	x21, x22, [sp, #32]
    1f54:	aa0003f4 	mov	x20, x0
	if(cap < len) {
		cap = len;
	}
	if(cap < 2) {
    1f58:	f100083f 	cmp	x1, #0x2
    1f5c:	d2800043 	mov	x3, #0x2                   	// #2
    1f60:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
    1f64:	eb02007f 	cmp	x3, x2
    1f68:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
		cap = 2;
	}
	vec->len = len;
    1f6c:	f9000001 	str	x1, [x0]
	vec->cap = cap;
    1f70:	f9000402 	str	x2, [x0, #8]
	vec->data = malloc(cap * sizeof *vec->data);
    1f74:	d37ef453 	lsl	x19, x2, #2
    1f78:	aa1303e0 	mov	x0, x19
    1f7c:	97fffca9 	bl	1220 <malloc@plt>
    1f80:	aa0003f5 	mov	x21, x0
    1f84:	f9000a80 	str	x0, [x20, #16]
	spt_CheckOSError(!vec->data, "ValVec New");
    1f88:	b4000140 	cbz	x0, 1fb0 <sptNewValueVector+0x6c>
	memset(vec->data, 0, cap * sizeof *vec->data);
    1f8c:	aa1303e2 	mov	x2, x19
    1f90:	52800001 	mov	w1, #0x0                   	// #0
    1f94:	aa1503e0 	mov	x0, x21
    1f98:	97fffcb6 	bl	1270 <memset@plt>
	return 0;
    1f9c:	52800000 	mov	w0, #0x0                   	// #0
}
    1fa0:	a94153f3 	ldp	x19, x20, [sp, #16]
    1fa4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1fa8:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1fac:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "ValVec New");
    1fb0:	97fffd04 	bl	13c0 <__errno_location@plt>
    1fb4:	aa0003f4 	mov	x20, x0
    1fb8:	b9400016 	ldr	w22, [x0]
    1fbc:	314042df 	cmn	w22, #0x10, lsl #12
    1fc0:	54fffe60 	b.eq	1f8c <sptNewValueVector+0x48>  // b.none
    1fc4:	2a1603e0 	mov	w0, w22
    1fc8:	97fffcbe 	bl	12c0 <strerror@plt>
    1fcc:	aa0003e4 	mov	x4, x0
    1fd0:	528005c3 	mov	w3, #0x2e                  	// #46
    1fd4:	f0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    1fd8:	91332042 	add	x2, x2, #0xcc8
    1fdc:	114042c1 	add	w1, w22, #0x10, lsl #12
    1fe0:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    1fe4:	9133c000 	add	x0, x0, #0xcf0
    1fe8:	94000961 	bl	456c <spt_ComplainError>
    1fec:	b9400280 	ldr	w0, [x20]
    1ff0:	11404000 	add	w0, w0, #0x10, lsl #12
    1ff4:	17ffffeb 	b	1fa0 <sptNewValueVector+0x5c>

0000000000001ff8 <sptConstantValueVector>:
 * @param val   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantValueVector(sptValueVector * const vec, sptValue const val) {
	for(sptNnzIndex i=0; i<vec->len; ++i)
    1ff8:	f9400001 	ldr	x1, [x0]
    1ffc:	b4000101 	cbz	x1, 201c <sptConstantValueVector+0x24>
    2000:	d2800001 	mov	x1, #0x0                   	// #0
		vec->data[i] = val;
    2004:	f9400802 	ldr	x2, [x0, #16]
    2008:	bc217840 	str	s0, [x2, x1, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    200c:	91000421 	add	x1, x1, #0x1
    2010:	f9400002 	ldr	x2, [x0]
    2014:	eb01005f 	cmp	x2, x1
    2018:	54ffff68 	b.hi	2004 <sptConstantValueVector+0xc>  // b.pmore
	return 0;
}
    201c:	52800000 	mov	w0, #0x0                   	// #0
    2020:	d65f03c0 	ret

0000000000002024 <sptAppendValueVector>:
 * @param vec   a pointer to a valid value vector
 * @param value the value to be appended
 *
 * The length of the value vector will be changed to contain the new value.
 */
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    2024:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2028:	910003fd 	mov	x29, sp
    202c:	a90153f3 	stp	x19, x20, [sp, #16]
    2030:	fd001fe8 	str	d8, [sp, #56]
    2034:	aa0003f3 	mov	x19, x0
    2038:	1e204008 	fmov	s8, s0
	if(vec->cap <= vec->len) {
    203c:	f9400400 	ldr	x0, [x0, #8]
    2040:	f9400261 	ldr	x1, [x19]
    2044:	eb01001f 	cmp	x0, x1
    2048:	54000189 	b.ls	2078 <sptAppendValueVector+0x54>  // b.plast
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
		spt_CheckOSError(!newdata, "ValVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
    204c:	f9400261 	ldr	x1, [x19]
    2050:	f9400a60 	ldr	x0, [x19, #16]
    2054:	bc217808 	str	s8, [x0, x1, lsl #2]
	++vec->len;
    2058:	f9400260 	ldr	x0, [x19]
    205c:	91000400 	add	x0, x0, #0x1
    2060:	f9000260 	str	x0, [x19]
	return 0;
    2064:	52800000 	mov	w0, #0x0                   	// #0
}
    2068:	fd401fe8 	ldr	d8, [sp, #56]
    206c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2070:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2074:	d65f03c0 	ret
    2078:	a9025bf5 	stp	x21, x22, [sp, #32]
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    207c:	8b400414 	add	x20, x0, x0, lsr #1
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2080:	d37ef681 	lsl	x1, x20, #2
    2084:	f9400a60 	ldr	x0, [x19, #16]
    2088:	97fffc7e 	bl	1280 <realloc@plt>
    208c:	aa0003f5 	mov	x21, x0
		spt_CheckOSError(!newdata, "ValVec Append");
    2090:	b40000a0 	cbz	x0, 20a4 <sptAppendValueVector+0x80>
		vec->cap = newcap;
    2094:	f9000674 	str	x20, [x19, #8]
		vec->data = newdata;
    2098:	f9000a75 	str	x21, [x19, #16]
    209c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    20a0:	17ffffeb 	b	204c <sptAppendValueVector+0x28>
    20a4:	f9001bf7 	str	x23, [sp, #48]
		spt_CheckOSError(!newdata, "ValVec Append");
    20a8:	97fffcc6 	bl	13c0 <__errno_location@plt>
    20ac:	aa0003f6 	mov	x22, x0
    20b0:	b9400017 	ldr	w23, [x0]
    20b4:	314042ff 	cmn	w23, #0x10, lsl #12
    20b8:	54000061 	b.ne	20c4 <sptAppendValueVector+0xa0>  // b.any
    20bc:	f9401bf7 	ldr	x23, [sp, #48]
    20c0:	17fffff5 	b	2094 <sptAppendValueVector+0x70>
    20c4:	2a1703e0 	mov	w0, w23
    20c8:	97fffc7e 	bl	12c0 <strerror@plt>
    20cc:	aa0003e4 	mov	x4, x0
    20d0:	52800a43 	mov	w3, #0x52                  	// #82
    20d4:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    20d8:	91332042 	add	x2, x2, #0xcc8
    20dc:	114042e1 	add	w1, w23, #0x10, lsl #12
    20e0:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    20e4:	91340000 	add	x0, x0, #0xd00
    20e8:	94000921 	bl	456c <spt_ComplainError>
    20ec:	b94002c0 	ldr	w0, [x22]
    20f0:	11404000 	add	w0, w0, #0x10, lsl #12
    20f4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    20f8:	f9401bf7 	ldr	x23, [sp, #48]
    20fc:	17ffffdb 	b	2068 <sptAppendValueVector+0x44>

0000000000002100 <sptResizeValueVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    2100:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2104:	910003fd 	mov	x29, sp
    2108:	a90153f3 	stp	x19, x20, [sp, #16]
    210c:	a9025bf5 	stp	x21, x22, [sp, #32]
    2110:	aa0003f3 	mov	x19, x0
    2114:	aa0103f4 	mov	x20, x1
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2118:	f100083f 	cmp	x1, #0x2
    211c:	d2800055 	mov	x21, #0x2                   	// #2
    2120:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    2124:	f9400400 	ldr	x0, [x0, #8]
    2128:	eb15001f 	cmp	x0, x21
    212c:	540000e1 	b.ne	2148 <sptResizeValueVector+0x48>  // b.any
		spt_CheckOSError(!newdata, "ValVec Resize");
		vec->len = size;
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
    2130:	f9000261 	str	x1, [x19]
	}
	return 0;
    2134:	52800000 	mov	w0, #0x0                   	// #0
}
    2138:	a94153f3 	ldp	x19, x20, [sp, #16]
    213c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2140:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2144:	d65f03c0 	ret
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2148:	d37ef6a1 	lsl	x1, x21, #2
    214c:	f9400a60 	ldr	x0, [x19, #16]
    2150:	97fffc4c 	bl	1280 <realloc@plt>
    2154:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "ValVec Resize");
    2158:	b40000c0 	cbz	x0, 2170 <sptResizeValueVector+0x70>
		vec->len = size;
    215c:	f9000274 	str	x20, [x19]
		vec->cap = newcap;
    2160:	f9000675 	str	x21, [x19, #8]
		vec->data = newdata;
    2164:	f9000a76 	str	x22, [x19, #16]
	return 0;
    2168:	52800000 	mov	w0, #0x0                   	// #0
    216c:	17fffff3 	b	2138 <sptResizeValueVector+0x38>
    2170:	a90363f7 	stp	x23, x24, [sp, #48]
		spt_CheckOSError(!newdata, "ValVec Resize");
    2174:	97fffc93 	bl	13c0 <__errno_location@plt>
    2178:	aa0003f7 	mov	x23, x0
    217c:	b9400018 	ldr	w24, [x0]
    2180:	3140431f 	cmn	w24, #0x10, lsl #12
    2184:	54000061 	b.ne	2190 <sptResizeValueVector+0x90>  // b.any
    2188:	a94363f7 	ldp	x23, x24, [sp, #48]
    218c:	17fffff4 	b	215c <sptResizeValueVector+0x5c>
    2190:	2a1803e0 	mov	w0, w24
    2194:	97fffc4b 	bl	12c0 <strerror@plt>
    2198:	aa0003e4 	mov	x4, x0
    219c:	52800d23 	mov	w3, #0x69                  	// #105
    21a0:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    21a4:	91332042 	add	x2, x2, #0xcc8
    21a8:	11404301 	add	w1, w24, #0x10, lsl #12
    21ac:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    21b0:	91344000 	add	x0, x0, #0xd10
    21b4:	940008ee 	bl	456c <spt_ComplainError>
    21b8:	b94002e0 	ldr	w0, [x23]
    21bc:	11404000 	add	w0, w0, #0x10, lsl #12
    21c0:	a94363f7 	ldp	x23, x24, [sp, #48]
    21c4:	17ffffdd 	b	2138 <sptResizeValueVector+0x38>

00000000000021c8 <sptFreeValueVector>:
 * Release the memory buffer a value vector is holding
 *
 * @param vec a pointer to a valid value vector
 *
 */
void sptFreeValueVector(sptValueVector *vec) {
    21c8:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    21cc:	910003fd 	mov	x29, sp
	vec->len = 0;
    21d0:	f900001f 	str	xzr, [x0]
	vec->cap = 0;
    21d4:	f900041f 	str	xzr, [x0, #8]
	free(vec->data);
    21d8:	f9400800 	ldr	x0, [x0, #16]
    21dc:	97fffc5d 	bl	1350 <free@plt>
}
    21e0:	a8c17bfd 	ldp	x29, x30, [sp], #16
    21e4:	d65f03c0 	ret

00000000000021e8 <sptNewIndexVector>:
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewIndexVector(sptIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    21e8:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    21ec:	910003fd 	mov	x29, sp
    21f0:	a90153f3 	stp	x19, x20, [sp, #16]
    21f4:	a9025bf5 	stp	x21, x22, [sp, #32]
    21f8:	aa0003f4 	mov	x20, x0
	if(cap < len) {
		cap = len;
	}
	if(cap < 2) {
    21fc:	f100083f 	cmp	x1, #0x2
    2200:	d2800043 	mov	x3, #0x2                   	// #2
    2204:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
    2208:	eb02007f 	cmp	x3, x2
    220c:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
		cap = 2;
	}
	vec->len = len;
    2210:	f9000001 	str	x1, [x0]
	vec->cap = cap;
    2214:	f9000402 	str	x2, [x0, #8]
	vec->data = malloc(cap * sizeof *vec->data);
    2218:	d37ef453 	lsl	x19, x2, #2
    221c:	aa1303e0 	mov	x0, x19
    2220:	97fffc00 	bl	1220 <malloc@plt>
    2224:	aa0003f5 	mov	x21, x0
    2228:	f9000a80 	str	x0, [x20, #16]
	spt_CheckOSError(!vec->data, "IdxVec New");
    222c:	b4000140 	cbz	x0, 2254 <sptNewIndexVector+0x6c>
	memset(vec->data, 0, cap * sizeof *vec->data);
    2230:	aa1303e2 	mov	x2, x19
    2234:	52800001 	mov	w1, #0x0                   	// #0
    2238:	aa1503e0 	mov	x0, x21
    223c:	97fffc0d 	bl	1270 <memset@plt>
	return 0;
    2240:	52800000 	mov	w0, #0x0                   	// #0
}
    2244:	a94153f3 	ldp	x19, x20, [sp, #16]
    2248:	a9425bf5 	ldp	x21, x22, [sp, #32]
    224c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2250:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "IdxVec New");
    2254:	97fffc5b 	bl	13c0 <__errno_location@plt>
    2258:	aa0003f4 	mov	x20, x0
    225c:	b9400016 	ldr	w22, [x0]
    2260:	314042df 	cmn	w22, #0x10, lsl #12
    2264:	54fffe60 	b.eq	2230 <sptNewIndexVector+0x48>  // b.none
    2268:	2a1603e0 	mov	w0, w22
    226c:	97fffc15 	bl	12c0 <strerror@plt>
    2270:	aa0003e4 	mov	x4, x0
    2274:	52801283 	mov	w3, #0x94                  	// #148
    2278:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    227c:	91332042 	add	x2, x2, #0xcc8
    2280:	114042c1 	add	w1, w22, #0x10, lsl #12
    2284:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2288:	91348000 	add	x0, x0, #0xd20
    228c:	940008b8 	bl	456c <spt_ComplainError>
    2290:	b9400280 	ldr	w0, [x20]
    2294:	11404000 	add	w0, w0, #0x10, lsl #12
    2298:	17ffffeb 	b	2244 <sptNewIndexVector+0x5c>

000000000000229c <sptAppendIndexVector>:
 * @param vec   a pointer to a valid index vector
 * @param value the value to be appended
 *
 * The length of the size vector will be changed to contain the new value.
 */
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    229c:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    22a0:	910003fd 	mov	x29, sp
    22a4:	a90153f3 	stp	x19, x20, [sp, #16]
    22a8:	a9025bf5 	stp	x21, x22, [sp, #32]
    22ac:	aa0003f3 	mov	x19, x0
    22b0:	2a0103f5 	mov	w21, w1
	if(vec->cap <= vec->len) {
    22b4:	f9400400 	ldr	x0, [x0, #8]
    22b8:	f9400262 	ldr	x2, [x19]
    22bc:	eb02001f 	cmp	x0, x2
    22c0:	54000189 	b.ls	22f0 <sptAppendIndexVector+0x54>  // b.plast
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
		spt_CheckOSError(!newdata, "IdxVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
    22c4:	f9400261 	ldr	x1, [x19]
    22c8:	f9400a60 	ldr	x0, [x19, #16]
    22cc:	b8217815 	str	w21, [x0, x1, lsl #2]
	++vec->len;
    22d0:	f9400260 	ldr	x0, [x19]
    22d4:	91000400 	add	x0, x0, #0x1
    22d8:	f9000260 	str	x0, [x19]
	return 0;
    22dc:	52800000 	mov	w0, #0x0                   	// #0
}
    22e0:	a94153f3 	ldp	x19, x20, [sp, #16]
    22e4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    22e8:	a8c47bfd 	ldp	x29, x30, [sp], #64
    22ec:	d65f03c0 	ret
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    22f0:	8b400414 	add	x20, x0, x0, lsr #1
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    22f4:	d37ef681 	lsl	x1, x20, #2
    22f8:	f9400a60 	ldr	x0, [x19, #16]
    22fc:	97fffbe1 	bl	1280 <realloc@plt>
    2300:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "IdxVec Append");
    2304:	b4000080 	cbz	x0, 2314 <sptAppendIndexVector+0x78>
		vec->cap = newcap;
    2308:	f9000674 	str	x20, [x19, #8]
		vec->data = newdata;
    230c:	f9000a76 	str	x22, [x19, #16]
    2310:	17ffffed 	b	22c4 <sptAppendIndexVector+0x28>
    2314:	a90363f7 	stp	x23, x24, [sp, #48]
		spt_CheckOSError(!newdata, "IdxVec Append");
    2318:	97fffc2a 	bl	13c0 <__errno_location@plt>
    231c:	aa0003f7 	mov	x23, x0
    2320:	b9400018 	ldr	w24, [x0]
    2324:	3140431f 	cmn	w24, #0x10, lsl #12
    2328:	54000061 	b.ne	2334 <sptAppendIndexVector+0x98>  // b.any
    232c:	a94363f7 	ldp	x23, x24, [sp, #48]
    2330:	17fffff6 	b	2308 <sptAppendIndexVector+0x6c>
    2334:	2a1803e0 	mov	w0, w24
    2338:	97fffbe2 	bl	12c0 <strerror@plt>
    233c:	aa0003e4 	mov	x4, x0
    2340:	52801543 	mov	w3, #0xaa                  	// #170
    2344:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2348:	91332042 	add	x2, x2, #0xcc8
    234c:	11404301 	add	w1, w24, #0x10, lsl #12
    2350:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2354:	9134c000 	add	x0, x0, #0xd30
    2358:	94000885 	bl	456c <spt_ComplainError>
    235c:	b94002e0 	ldr	w0, [x23]
    2360:	11404000 	add	w0, w0, #0x10, lsl #12
    2364:	a94363f7 	ldp	x23, x24, [sp, #48]
    2368:	17ffffde 	b	22e0 <sptAppendIndexVector+0x44>

000000000000236c <sptResizeIndexVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    236c:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2370:	910003fd 	mov	x29, sp
    2374:	a90153f3 	stp	x19, x20, [sp, #16]
    2378:	a9025bf5 	stp	x21, x22, [sp, #32]
    237c:	aa0003f3 	mov	x19, x0
    2380:	aa0103f4 	mov	x20, x1
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2384:	f100083f 	cmp	x1, #0x2
    2388:	d2800055 	mov	x21, #0x2                   	// #2
    238c:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    2390:	f9400400 	ldr	x0, [x0, #8]
    2394:	eb15001f 	cmp	x0, x21
    2398:	540000e1 	b.ne	23b4 <sptResizeIndexVector+0x48>  // b.any
		spt_CheckOSError(!newdata, "IdxVec Resize");
		vec->len = size;
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
    239c:	f9000261 	str	x1, [x19]
	}
	return 0;
    23a0:	52800000 	mov	w0, #0x0                   	// #0
}
    23a4:	a94153f3 	ldp	x19, x20, [sp, #16]
    23a8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    23ac:	a8c47bfd 	ldp	x29, x30, [sp], #64
    23b0:	d65f03c0 	ret
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    23b4:	d37ef6a1 	lsl	x1, x21, #2
    23b8:	f9400a60 	ldr	x0, [x19, #16]
    23bc:	97fffbb1 	bl	1280 <realloc@plt>
    23c0:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "IdxVec Resize");
    23c4:	b40000c0 	cbz	x0, 23dc <sptResizeIndexVector+0x70>
		vec->len = size;
    23c8:	f9000274 	str	x20, [x19]
		vec->cap = newcap;
    23cc:	f9000675 	str	x21, [x19, #8]
		vec->data = newdata;
    23d0:	f9000a76 	str	x22, [x19, #16]
	return 0;
    23d4:	52800000 	mov	w0, #0x0                   	// #0
    23d8:	17fffff3 	b	23a4 <sptResizeIndexVector+0x38>
    23dc:	a90363f7 	stp	x23, x24, [sp, #48]
		spt_CheckOSError(!newdata, "IdxVec Resize");
    23e0:	97fffbf8 	bl	13c0 <__errno_location@plt>
    23e4:	aa0003f7 	mov	x23, x0
    23e8:	b9400018 	ldr	w24, [x0]
    23ec:	3140431f 	cmn	w24, #0x10, lsl #12
    23f0:	54000061 	b.ne	23fc <sptResizeIndexVector+0x90>  // b.any
    23f4:	a94363f7 	ldp	x23, x24, [sp, #48]
    23f8:	17fffff4 	b	23c8 <sptResizeIndexVector+0x5c>
    23fc:	2a1803e0 	mov	w0, w24
    2400:	97fffbb0 	bl	12c0 <strerror@plt>
    2404:	aa0003e4 	mov	x4, x0
    2408:	52801823 	mov	w3, #0xc1                  	// #193
    240c:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2410:	91332042 	add	x2, x2, #0xcc8
    2414:	11404301 	add	w1, w24, #0x10, lsl #12
    2418:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    241c:	91350000 	add	x0, x0, #0xd40
    2420:	94000853 	bl	456c <spt_ComplainError>
    2424:	b94002e0 	ldr	w0, [x23]
    2428:	11404000 	add	w0, w0, #0x10, lsl #12
    242c:	a94363f7 	ldp	x23, x24, [sp, #48]
    2430:	17ffffdd 	b	23a4 <sptResizeIndexVector+0x38>

0000000000002434 <sptFreeIndexVector>:
 * Release the memory buffer a sptIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void sptFreeIndexVector(sptIndexVector *vec) {
    2434:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2438:	910003fd 	mov	x29, sp
    243c:	f9000bf3 	str	x19, [sp, #16]
    2440:	aa0003f3 	mov	x19, x0
	free(vec->data);
    2444:	f9400800 	ldr	x0, [x0, #16]
    2448:	97fffbc2 	bl	1350 <free@plt>
	vec->len = 0;
    244c:	f900027f 	str	xzr, [x19]
	vec->cap = 0;
    2450:	f900067f 	str	xzr, [x19, #8]
}
    2454:	f9400bf3 	ldr	x19, [sp, #16]
    2458:	a8c27bfd 	ldp	x29, x30, [sp], #32
    245c:	d65f03c0 	ret

0000000000002460 <fill_binary_nnzidx._omp_fn.0>:
		sptIndex const BUF_LEN = 1024*1024;
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    2460:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2464:	910003fd 	mov	x29, sp
    2468:	f90013f5 	str	x21, [sp, #32]
    246c:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    2470:	34000355 	cbz	w21, 24d8 <fill_binary_nnzidx._omp_fn.0+0x78>
    2474:	a90153f3 	stp	x19, x20, [sp, #16]
    2478:	aa0003f4 	mov	x20, x0
    247c:	97fffb8d 	bl	12b0 <omp_get_num_threads@plt>
    2480:	2a0003f3 	mov	w19, w0
    2484:	97fffb57 	bl	11e0 <omp_get_thread_num@plt>
    2488:	1ad30aa2 	udiv	w2, w21, w19
    248c:	1b13d441 	msub	w1, w2, w19, w21
    2490:	6b01001f 	cmp	w0, w1
    2494:	54000283 	b.cc	24e4 <fill_binary_nnzidx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    2498:	1b000441 	madd	w1, w2, w0, w1
    249c:	0b010043 	add	w3, w2, w1
    24a0:	6b03003f 	cmp	w1, w3
    24a4:	54000262 	b.cs	24f0 <fill_binary_nnzidx._omp_fn.0+0x90>  // b.hs, b.nlast
#pragma omp parallel for schedule(static)
    24a8:	b9401284 	ldr	w4, [x20, #16]
    24ac:	f9400285 	ldr	x5, [x20]
    24b0:	f9400682 	ldr	x2, [x20, #8]
    24b4:	8b214842 	add	x2, x2, w1, uxtw #2
    24b8:	0b010080 	add	w0, w4, w1
    24bc:	0b030083 	add	w3, w4, w3
				buffer[n + i] = ubuf[i];
    24c0:	b8404441 	ldr	w1, [x2], #4
    24c4:	f82058a1 	str	x1, [x5, w0, uxtw #3]
    24c8:	11000400 	add	w0, w0, #0x1
    24cc:	6b03001f 	cmp	w0, w3
    24d0:	54ffff81 	b.ne	24c0 <fill_binary_nnzidx._omp_fn.0+0x60>  // b.any
    24d4:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    24d8:	f94013f5 	ldr	x21, [sp, #32]
    24dc:	a8c37bfd 	ldp	x29, x30, [sp], #48
    24e0:	d65f03c0 	ret
    24e4:	11000442 	add	w2, w2, #0x1
    24e8:	52800001 	mov	w1, #0x0                   	// #0
    24ec:	17ffffeb 	b	2498 <fill_binary_nnzidx._omp_fn.0+0x38>
    24f0:	a94153f3 	ldp	x19, x20, [sp, #16]
    24f4:	17fffff9 	b	24d8 <fill_binary_nnzidx._omp_fn.0+0x78>

00000000000024f8 <fill_binary_idx>:
{
    24f8:	a9b87bfd 	stp	x29, x30, [sp, #-128]!
    24fc:	910003fd 	mov	x29, sp
    2500:	a9025bf5 	stp	x21, x22, [sp, #32]
    2504:	a90363f7 	stp	x23, x24, [sp, #48]
    2508:	aa0003f7 	mov	x23, x0
    250c:	2a0103f5 	mov	w21, w1
    2510:	aa0303f8 	mov	x24, x3
	if(header->idx_width == sizeof(sptIndex)) {
    2514:	f9400440 	ldr	x0, [x2, #8]
    2518:	f100101f 	cmp	x0, #0x4
    251c:	54000560 	b.eq	25c8 <fill_binary_idx+0xd0>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2520:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2524:	97fffb3f 	bl	1220 <malloc@plt>
    2528:	aa0003f6 	mov	x22, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    252c:	34000435 	cbz	w21, 25b0 <fill_binary_idx+0xb8>
    2530:	a90153f3 	stp	x19, x20, [sp, #16]
    2534:	a9046bf9 	stp	x25, x26, [sp, #64]
    2538:	a90573fb 	stp	x27, x28, [sp, #80]
    253c:	52800014 	mov	w20, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2540:	52a0021c 	mov	w28, #0x100000              	// #1048576
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2544:	d280009b 	mov	x27, #0x4                   	// #4
    2548:	9101a3fa 	add	x26, sp, #0x68
    254c:	90000019 	adrp	x25, 2000 <sptConstantValueVector+0x8>
    2550:	91177339 	add	x25, x25, #0x5dc
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2554:	4b1402b3 	sub	w19, w21, w20
    2558:	7144027f 	cmp	w19, #0x100, lsl #12
    255c:	1a9c9273 	csel	w19, w19, w28, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2560:	aa1803e3 	mov	x3, x24
    2564:	2a1303e2 	mov	w2, w19
    2568:	aa1b03e1 	mov	x1, x27
    256c:	aa1603e0 	mov	x0, x22
    2570:	97fffb74 	bl	1340 <fread@plt>
#pragma omp parallel for schedule(static)
    2574:	b9007ff3 	str	w19, [sp, #124]
    2578:	b9007bf4 	str	w20, [sp, #120]
    257c:	f9003bf6 	str	x22, [sp, #112]
    2580:	f90037f7 	str	x23, [sp, #104]
    2584:	52800003 	mov	w3, #0x0                   	// #0
    2588:	52800002 	mov	w2, #0x0                   	// #0
    258c:	aa1a03e1 	mov	x1, x26
    2590:	aa1903e0 	mov	x0, x25
    2594:	97fffb8f 	bl	13d0 <GOMP_parallel@plt>
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2598:	11440294 	add	w20, w20, #0x100, lsl #12
    259c:	6b1402bf 	cmp	w21, w20
    25a0:	54fffda8 	b.hi	2554 <fill_binary_idx+0x5c>  // b.pmore
    25a4:	a94153f3 	ldp	x19, x20, [sp, #16]
    25a8:	a9446bf9 	ldp	x25, x26, [sp, #64]
    25ac:	a94573fb 	ldp	x27, x28, [sp, #80]
		free(ubuf);
    25b0:	aa1603e0 	mov	x0, x22
    25b4:	97fffb67 	bl	1350 <free@plt>
}
    25b8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    25bc:	a94363f7 	ldp	x23, x24, [sp, #48]
    25c0:	a8c87bfd 	ldp	x29, x30, [sp], #128
    25c4:	d65f03c0 	ret
		fread(buffer, sizeof(sptIndex), count, fin);
    25c8:	2a0103e2 	mov	w2, w1
    25cc:	d2800081 	mov	x1, #0x4                   	// #4
    25d0:	aa1703e0 	mov	x0, x23
    25d4:	97fffb5b 	bl	1340 <fread@plt>
    25d8:	17fffff8 	b	25b8 <fill_binary_idx+0xc0>

00000000000025dc <fill_binary_idx._omp_fn.0>:
#pragma omp parallel for schedule(static)
    25dc:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    25e0:	910003fd 	mov	x29, sp
    25e4:	f90013f5 	str	x21, [sp, #32]
    25e8:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    25ec:	34000355 	cbz	w21, 2654 <fill_binary_idx._omp_fn.0+0x78>
    25f0:	a90153f3 	stp	x19, x20, [sp, #16]
    25f4:	aa0003f4 	mov	x20, x0
    25f8:	97fffb2e 	bl	12b0 <omp_get_num_threads@plt>
    25fc:	2a0003f3 	mov	w19, w0
    2600:	97fffaf8 	bl	11e0 <omp_get_thread_num@plt>
    2604:	1ad30aa1 	udiv	w1, w21, w19
    2608:	1b13d422 	msub	w2, w1, w19, w21
    260c:	6b02001f 	cmp	w0, w2
    2610:	54000283 	b.cc	2660 <fill_binary_idx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    2614:	1b000822 	madd	w2, w1, w0, w2
    2618:	0b020023 	add	w3, w1, w2
    261c:	6b03005f 	cmp	w2, w3
    2620:	54000262 	b.cs	266c <fill_binary_idx._omp_fn.0+0x90>  // b.hs, b.nlast
#pragma omp parallel for schedule(static)
    2624:	b9401284 	ldr	w4, [x20, #16]
    2628:	f9400285 	ldr	x5, [x20]
    262c:	0b020081 	add	w1, w4, w2
    2630:	f9400680 	ldr	x0, [x20, #8]
    2634:	8b224800 	add	x0, x0, w2, uxtw #2
    2638:	0b030083 	add	w3, w4, w3
				buffer[n + i] = ubuf[i];
    263c:	b8404402 	ldr	w2, [x0], #4
    2640:	b82158a2 	str	w2, [x5, w1, uxtw #2]
    2644:	11000421 	add	w1, w1, #0x1
    2648:	6b03003f 	cmp	w1, w3
    264c:	54ffff81 	b.ne	263c <fill_binary_idx._omp_fn.0+0x60>  // b.any
    2650:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    2654:	f94013f5 	ldr	x21, [sp, #32]
    2658:	a8c37bfd 	ldp	x29, x30, [sp], #48
    265c:	d65f03c0 	ret
    2660:	11000421 	add	w1, w1, #0x1
    2664:	52800002 	mov	w2, #0x0                   	// #0
    2668:	17ffffeb 	b	2614 <fill_binary_idx._omp_fn.0+0x38>
    266c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2670:	17fffff9 	b	2654 <fill_binary_idx._omp_fn.0+0x78>

0000000000002674 <fill_binary_val._omp_fn.0>:
#endif

		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    2674:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2678:	910003fd 	mov	x29, sp
    267c:	f90013f5 	str	x21, [sp, #32]
    2680:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    2684:	34000375 	cbz	w21, 26f0 <fill_binary_val._omp_fn.0+0x7c>
    2688:	a90153f3 	stp	x19, x20, [sp, #16]
    268c:	aa0003f4 	mov	x20, x0
    2690:	97fffb08 	bl	12b0 <omp_get_num_threads@plt>
    2694:	2a0003f3 	mov	w19, w0
    2698:	97fffad2 	bl	11e0 <omp_get_thread_num@plt>
    269c:	1ad30aa2 	udiv	w2, w21, w19
    26a0:	1b13d441 	msub	w1, w2, w19, w21
    26a4:	6b01001f 	cmp	w0, w1
    26a8:	540002a3 	b.cc	26fc <fill_binary_val._omp_fn.0+0x88>  // b.lo, b.ul, b.last
    26ac:	1b000441 	madd	w1, w2, w0, w1
    26b0:	0b010043 	add	w3, w2, w1
    26b4:	6b03003f 	cmp	w1, w3
    26b8:	54000282 	b.cs	2708 <fill_binary_val._omp_fn.0+0x94>  // b.hs, b.nlast
#pragma omp parallel for schedule(static)
    26bc:	b9401284 	ldr	w4, [x20, #16]
    26c0:	f9400285 	ldr	x5, [x20]
    26c4:	f9400682 	ldr	x2, [x20, #8]
    26c8:	8b214c42 	add	x2, x2, w1, uxtw #3
    26cc:	0b010080 	add	w0, w4, w1
    26d0:	0b030083 	add	w3, w4, w3
				buffer[n + i] = ubuf[i];
    26d4:	fc408440 	ldr	d0, [x2], #8
    26d8:	1e624000 	fcvt	s0, d0
    26dc:	bc2058a0 	str	s0, [x5, w0, uxtw #2]
    26e0:	11000400 	add	w0, w0, #0x1
    26e4:	6b03001f 	cmp	w0, w3
    26e8:	54ffff61 	b.ne	26d4 <fill_binary_val._omp_fn.0+0x60>  // b.any
    26ec:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    26f0:	f94013f5 	ldr	x21, [sp, #32]
    26f4:	a8c37bfd 	ldp	x29, x30, [sp], #48
    26f8:	d65f03c0 	ret
    26fc:	11000442 	add	w2, w2, #0x1
    2700:	52800001 	mov	w1, #0x0                   	// #0
    2704:	17ffffea 	b	26ac <fill_binary_val._omp_fn.0+0x38>
    2708:	a94153f3 	ldp	x19, x20, [sp, #16]
    270c:	17fffff9 	b	26f0 <fill_binary_val._omp_fn.0+0x7c>

0000000000002710 <sptLoadSparseTensor>:
 * @param tsr         th sparse tensor to store into
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to read from
 */
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, char const * const fname)
{
    2710:	a9b47bfd 	stp	x29, x30, [sp, #-192]!
    2714:	910003fd 	mov	x29, sp
    2718:	a90153f3 	stp	x19, x20, [sp, #16]
    271c:	a90363f7 	stp	x23, x24, [sp, #48]
    2720:	a9046bf9 	stp	x25, x26, [sp, #64]
    2724:	aa0003f3 	mov	x19, x0
    2728:	2a0103f7 	mov	w23, w1
    272c:	aa0203f9 	mov	x25, x2
	FILE * fp = fopen(fname, "r");
    2730:	d0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    2734:	912d4021 	add	x1, x1, #0xb50
    2738:	aa0203e0 	mov	x0, x2
    273c:	97fffab1 	bl	1200 <fopen@plt>
	sptAssert(fp != NULL);
    2740:	b4000a80 	cbz	x0, 2890 <sptLoadSparseTensor+0x180>
    2744:	a9025bf5 	stp	x21, x22, [sp, #32]
    2748:	aa0003f6 	mov	x22, x0
	char const * const suffix = strrchr(fname, '.');
    274c:	528005c1 	mov	w1, #0x2e                  	// #46
    2750:	aa1903e0 	mov	x0, x25
    2754:	97fffadf 	bl	12d0 <strrchr@plt>
    2758:	aa0003f8 	mov	x24, x0
	if(suffix == NULL) {
    275c:	b4000160 	cbz	x0, 2788 <sptLoadSparseTensor+0x78>
    2760:	900000b4 	adrp	x20, 16000 <__FRAME_END__+0x100e8>
    2764:	91350294 	add	x20, x20, #0xd40
	size_t idx = 0;
    2768:	d2800015 	mov	x21, #0x0                   	// #0
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    276c:	f9400281 	ldr	x1, [x20]
    2770:	aa1803e0 	mov	x0, x24
    2774:	97fffaef 	bl	1330 <strcmp@plt>
    2778:	34000940 	cbz	w0, 28a0 <sptLoadSparseTensor+0x190>
	} while(file_extensions[++idx].extension != NULL);
    277c:	910006b5 	add	x21, x21, #0x1
    2780:	f8410e83 	ldr	x3, [x20, #16]!
    2784:	b5ffff43 	cbnz	x3, 276c <sptLoadSparseTensor+0x5c>
	fprintf(stderr, "SPLATT: extension for '%s' not recognized. "
    2788:	aa1903e2 	mov	x2, x25
    278c:	d0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    2790:	91354021 	add	x1, x1, #0xd50
    2794:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    2798:	f947d800 	ldr	x0, [x0, #4016]
    279c:	f9400000 	ldr	x0, [x0]
    27a0:	97fffb14 	bl	13f0 <fprintf@plt>
	iores = fscanf(fp, "%u", &tsr->nmodes);
    27a4:	aa1303e2 	mov	x2, x19
    27a8:	d0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    27ac:	912ac021 	add	x1, x1, #0xab0
    27b0:	aa1603e0 	mov	x0, x22
    27b4:	97fffa9f 	bl	1230 <__isoc99_fscanf@plt>
	spt_CheckOSError(iores < 0, "SpTns Load");
    27b8:	37f81a00 	tbnz	w0, #31, 2af8 <sptLoadSparseTensor+0x3e8>
	tsr->sortorder = malloc(tsr->nmodes * sizeof tsr->sortorder[0]);
    27bc:	b9400275 	ldr	w21, [x19]
    27c0:	d37e7eb8 	ubfiz	x24, x21, #2, #32
    27c4:	aa1803e0 	mov	x0, x24
    27c8:	97fffa96 	bl	1220 <malloc@plt>
    27cc:	aa0003f4 	mov	x20, x0
    27d0:	f9000660 	str	x0, [x19, #8]
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    27d4:	b4001b60 	cbz	x0, 2b40 <sptLoadSparseTensor+0x430>
	memset(tsr->sortorder, 0, tsr->nmodes * sizeof tsr->sortorder[0]);
    27d8:	aa1803e2 	mov	x2, x24
    27dc:	52800001 	mov	w1, #0x0                   	// #0
    27e0:	aa1403e0 	mov	x0, x20
    27e4:	97fffaa3 	bl	1270 <memset@plt>
	tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    27e8:	aa1803e0 	mov	x0, x24
    27ec:	97fffa8d 	bl	1220 <malloc@plt>
    27f0:	f9000a60 	str	x0, [x19, #16]
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    27f4:	b4001ca0 	cbz	x0, 2b88 <sptLoadSparseTensor+0x478>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    27f8:	52800014 	mov	w20, #0x0                   	// #0
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    27fc:	d0000018 	adrp	x24, 4000 <sptOmpMTTKRP+0x13c>
    2800:	912ac318 	add	x24, x24, #0xab0
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2804:	35001ef5 	cbnz	w21, 2be0 <sptLoadSparseTensor+0x4d0>
	tsr->nnz = 0;
    2808:	f9000e7f 	str	xzr, [x19, #24]
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    280c:	d37f7ea0 	ubfiz	x0, x21, #1, #32
    2810:	8b354000 	add	x0, x0, w21, uxtw
    2814:	d37df000 	lsl	x0, x0, #3
    2818:	97fffa82 	bl	1220 <malloc@plt>
    281c:	f9001260 	str	x0, [x19, #32]
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2820:	b4002240 	cbz	x0, 2c68 <sptLoadSparseTensor+0x558>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2824:	b9400260 	ldr	w0, [x19]
    2828:	52800014 	mov	w20, #0x0                   	// #0
    282c:	340001a0 	cbz	w0, 2860 <sptLoadSparseTensor+0x150>
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    2830:	d37f7e80 	ubfiz	x0, x20, #1, #32
    2834:	8b344000 	add	x0, x0, w20, uxtw
    2838:	f9401263 	ldr	x3, [x19, #32]
    283c:	d2800002 	mov	x2, #0x0                   	// #0
    2840:	d2800001 	mov	x1, #0x0                   	// #0
    2844:	8b000c60 	add	x0, x3, x0, lsl #3
    2848:	97fffe68 	bl	21e8 <sptNewIndexVector>
		spt_CheckError(retval, "SpTns Load", NULL);
    284c:	35002320 	cbnz	w0, 2cb0 <sptLoadSparseTensor+0x5a0>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2850:	11000694 	add	w20, w20, #0x1
    2854:	b9400260 	ldr	w0, [x19]
    2858:	6b00029f 	cmp	w20, w0
    285c:	54fffea3 	b.cc	2830 <sptLoadSparseTensor+0x120>  // b.lo, b.ul, b.last
	retval = sptNewValueVector(&tsr->values, 0, 0);
    2860:	9100a279 	add	x25, x19, #0x28
    2864:	d2800002 	mov	x2, #0x0                   	// #0
    2868:	d2800001 	mov	x1, #0x0                   	// #0
    286c:	aa1903e0 	mov	x0, x25
    2870:	97fffdb5 	bl	1f44 <sptNewValueVector>
	spt_CheckError(retval, "SpTns Load", NULL);
    2874:	35002520 	cbnz	w0, 2d18 <sptLoadSparseTensor+0x608>
			iores = fscanf(fp, "%u", &index);
    2878:	9101e3f8 	add	x24, sp, #0x78
    287c:	d0000015 	adrp	x21, 4000 <sptOmpMTTKRP+0x13c>
    2880:	912ac2b5 	add	x21, x21, #0xab0
			iores = fscanf(fp, "%lf", &value);
    2884:	d000001a 	adrp	x26, 4000 <sptOmpMTTKRP+0x13c>
    2888:	9137e35a 	add	x26, x26, #0xdf8
    288c:	14000143 	b	2d98 <sptLoadSparseTensor+0x688>
    2890:	a9025bf5 	stp	x21, x22, [sp, #32]
    2894:	a90573fb 	stp	x27, x28, [sp, #80]
	sptAssert(fp != NULL);
    2898:	12800000 	mov	w0, #0xffffffff            	// #-1
    289c:	97fffa39 	bl	1180 <exit@plt>
			return file_extensions[idx].type;
    28a0:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    28a4:	91350000 	add	x0, x0, #0xd40
    28a8:	8b151015 	add	x21, x0, x21, lsl #4
    28ac:	b9400aa0 	ldr	w0, [x21, #8]

	int iores;
	switch(get_file_type(fname)) {
    28b0:	34fff7a0 	cbz	w0, 27a4 <sptLoadSparseTensor+0x94>
    28b4:	7100041f 	cmp	w0, #0x1
    28b8:	54001c61 	b.ne	2c44 <sptLoadSparseTensor+0x534>  // b.any
	fread(&(header->magic), sizeof(header->magic), 1, fin);
    28bc:	aa1603e3 	mov	x3, x22
    28c0:	d2800022 	mov	x2, #0x1                   	// #1
    28c4:	d2800081 	mov	x1, #0x4                   	// #4
    28c8:	9101e3e0 	add	x0, sp, #0x78
    28cc:	97fffa9d 	bl	1340 <fread@plt>
	fread(&(header->idx_width), sizeof(header->idx_width), 1, fin);
    28d0:	aa1603e3 	mov	x3, x22
    28d4:	d2800022 	mov	x2, #0x1                   	// #1
    28d8:	d2800101 	mov	x1, #0x8                   	// #8
    28dc:	910203e0 	add	x0, sp, #0x80
    28e0:	97fffa98 	bl	1340 <fread@plt>
	fread(&(header->val_width), sizeof(header->val_width), 1, fin);
    28e4:	aa1603e3 	mov	x3, x22
    28e8:	d2800022 	mov	x2, #0x1                   	// #1
    28ec:	d2800101 	mov	x1, #0x8                   	// #8
    28f0:	910223e0 	add	x0, sp, #0x88
    28f4:	97fffa93 	bl	1340 <fread@plt>
	if(header->idx_width > PASTA_INDEX_TYPEWIDTH / 8) {
    28f8:	f94043e0 	ldr	x0, [sp, #128]
    28fc:	f100101f 	cmp	x0, #0x4
    2900:	54002948 	b.hi	2e28 <sptLoadSparseTensor+0x718>  // b.pmore
	if(header->val_width > PASTA_VALUE_TYPEWIDTH / 8) {
    2904:	f94047e0 	ldr	x0, [sp, #136]
    2908:	f100101f 	cmp	x0, #0x4
    290c:	54002a48 	b.hi	2e54 <sptLoadSparseTensor+0x744>  // b.pmore
	sptNnzIndex nnz = 0;
    2910:	f9003bff 	str	xzr, [sp, #112]
	sptIndex nmodes = 0;
    2914:	b9006fff 	str	wzr, [sp, #108]
	fill_binary_idx(&nmodes, 1, &header, fin);
    2918:	9101e3f4 	add	x20, sp, #0x78
    291c:	aa1603e3 	mov	x3, x22
    2920:	aa1403e2 	mov	x2, x20
    2924:	52800021 	mov	w1, #0x1                   	// #1
    2928:	9101b3e0 	add	x0, sp, #0x6c
    292c:	97fffef3 	bl	24f8 <fill_binary_idx>
	sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
    2930:	b9406ff7 	ldr	w23, [sp, #108]
    2934:	d37e7ee0 	ubfiz	x0, x23, #2, #32
    2938:	97fffa3a 	bl	1220 <malloc@plt>
    293c:	aa0003f5 	mov	x21, x0
	fill_binary_idx(dims, nmodes, &header, fin);
    2940:	aa1603e3 	mov	x3, x22
    2944:	aa1403e2 	mov	x2, x20
    2948:	2a1703e1 	mov	w1, w23
    294c:	97fffeeb 	bl	24f8 <fill_binary_idx>
	if(header->idx_width == sizeof(sptNnzIndex)) {
    2950:	f94043e0 	ldr	x0, [sp, #128]
    2954:	f100201f 	cmp	x0, #0x8
    2958:	54002900 	b.eq	2e78 <sptLoadSparseTensor+0x768>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    295c:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2960:	97fffa30 	bl	1220 <malloc@plt>
    2964:	aa0003f4 	mov	x20, x0
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2968:	aa1603e3 	mov	x3, x22
    296c:	d2800022 	mov	x2, #0x1                   	// #1
    2970:	d2800081 	mov	x1, #0x4                   	// #4
    2974:	97fffa73 	bl	1340 <fread@plt>
#pragma omp parallel for schedule(static)
    2978:	52800020 	mov	w0, #0x1                   	// #1
    297c:	b900a7e0 	str	w0, [sp, #164]
    2980:	b900a3ff 	str	wzr, [sp, #160]
    2984:	f9004ff4 	str	x20, [sp, #152]
    2988:	9101c3e0 	add	x0, sp, #0x70
    298c:	f9004be0 	str	x0, [sp, #144]
    2990:	52800003 	mov	w3, #0x0                   	// #0
    2994:	52800002 	mov	w2, #0x0                   	// #0
    2998:	910243e1 	add	x1, sp, #0x90
    299c:	90000000 	adrp	x0, 2000 <sptConstantValueVector+0x8>
    29a0:	91118000 	add	x0, x0, #0x460
    29a4:	97fffa8b 	bl	13d0 <GOMP_parallel@plt>
		free(ubuf);
    29a8:	aa1403e0 	mov	x0, x20
    29ac:	97fffa69 	bl	1350 <free@plt>
	sptNewSparseTensor(tsr, nmodes, dims);
    29b0:	aa1503e2 	mov	x2, x21
    29b4:	b9406fe1 	ldr	w1, [sp, #108]
    29b8:	aa1303e0 	mov	x0, x19
    29bc:	97fffcf0 	bl	1d7c <sptNewSparseTensor>
	tsr->nnz = nnz;
    29c0:	f9403be0 	ldr	x0, [sp, #112]
    29c4:	f9000e60 	str	x0, [x19, #24]
	for(sptIndex m=0; m < nmodes; ++m) {
    29c8:	b9406fe0 	ldr	w0, [sp, #108]
    29cc:	340001a0 	cbz	w0, 2a00 <sptLoadSparseTensor+0x2f0>
    29d0:	52800014 	mov	w20, #0x0                   	// #0
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    29d4:	d37f7e80 	ubfiz	x0, x20, #1, #32
    29d8:	8b344000 	add	x0, x0, w20, uxtw
    29dc:	f9401262 	ldr	x2, [x19, #32]
    29e0:	f9403be1 	ldr	x1, [sp, #112]
    29e4:	8b000c40 	add	x0, x2, x0, lsl #3
    29e8:	97fffe61 	bl	236c <sptResizeIndexVector>
		spt_CheckError(result, "SpTns Read", NULL);
    29ec:	35002520 	cbnz	w0, 2e90 <sptLoadSparseTensor+0x780>
	for(sptIndex m=0; m < nmodes; ++m) {
    29f0:	11000694 	add	w20, w20, #0x1
    29f4:	b9406fe0 	ldr	w0, [sp, #108]
    29f8:	6b00029f 	cmp	w20, w0
    29fc:	54fffec3 	b.cc	29d4 <sptLoadSparseTensor+0x2c4>  // b.lo, b.ul, b.last
	result = sptResizeValueVector(&tsr->values, nnz);
    2a00:	f9403be1 	ldr	x1, [sp, #112]
    2a04:	9100a260 	add	x0, x19, #0x28
    2a08:	97fffdbe 	bl	2100 <sptResizeValueVector>
	spt_CheckError(result, "SpTns Read", NULL);
    2a0c:	35002760 	cbnz	w0, 2ef8 <sptLoadSparseTensor+0x7e8>
	for(sptIndex m=0; m < nmodes; ++m) {
    2a10:	b9406fe0 	ldr	w0, [sp, #108]
    2a14:	52800014 	mov	w20, #0x0                   	// #0
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2a18:	9101e3f5 	add	x21, sp, #0x78
	for(sptIndex m=0; m < nmodes; ++m) {
    2a1c:	340001c0 	cbz	w0, 2a54 <sptLoadSparseTensor+0x344>
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2a20:	f9401261 	ldr	x1, [x19, #32]
    2a24:	d37f7e80 	ubfiz	x0, x20, #1, #32
    2a28:	8b344000 	add	x0, x0, w20, uxtw
    2a2c:	8b000c20 	add	x0, x1, x0, lsl #3
    2a30:	aa1603e3 	mov	x3, x22
    2a34:	aa1503e2 	mov	x2, x21
    2a38:	b94073e1 	ldr	w1, [sp, #112]
    2a3c:	f9400800 	ldr	x0, [x0, #16]
    2a40:	97fffeae 	bl	24f8 <fill_binary_idx>
	for(sptIndex m=0; m < nmodes; ++m) {
    2a44:	11000694 	add	w20, w20, #0x1
    2a48:	b9406fe0 	ldr	w0, [sp, #108]
    2a4c:	6b00029f 	cmp	w20, w0
    2a50:	54fffe83 	b.cc	2a20 <sptLoadSparseTensor+0x310>  // b.lo, b.ul, b.last
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2a54:	f9401e79 	ldr	x25, [x19, #56]
    2a58:	f9403bf3 	ldr	x19, [sp, #112]
	if(header->val_width == sizeof(sptValue)) {
    2a5c:	f94047e0 	ldr	x0, [sp, #136]
    2a60:	f100101f 	cmp	x0, #0x4
    2a64:	540025c0 	b.eq	2f1c <sptLoadSparseTensor+0x80c>  // b.none
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2a68:	2a1303f7 	mov	w23, w19
		double * ubuf = (double*)malloc(BUF_LEN * sizeof(*ubuf));
    2a6c:	d2a01000 	mov	x0, #0x800000              	// #8388608
    2a70:	97fff9ec 	bl	1220 <malloc@plt>
    2a74:	aa0003f5 	mov	x21, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2a78:	340003b3 	cbz	w19, 2aec <sptLoadSparseTensor+0x3dc>
    2a7c:	a90573fb 	stp	x27, x28, [sp, #80]
    2a80:	52800014 	mov	w20, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2a84:	52a0021c 	mov	w28, #0x100000              	// #1048576
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2a88:	d280011b 	mov	x27, #0x8                   	// #8
    2a8c:	9102a3fa 	add	x26, sp, #0xa8
    2a90:	90000018 	adrp	x24, 2000 <sptConstantValueVector+0x8>
    2a94:	9119d318 	add	x24, x24, #0x674
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2a98:	4b1402f3 	sub	w19, w23, w20
    2a9c:	7144027f 	cmp	w19, #0x100, lsl #12
    2aa0:	1a9c9273 	csel	w19, w19, w28, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2aa4:	aa1603e3 	mov	x3, x22
    2aa8:	2a1303e2 	mov	w2, w19
    2aac:	aa1b03e1 	mov	x1, x27
    2ab0:	aa1503e0 	mov	x0, x21
    2ab4:	97fffa23 	bl	1340 <fread@plt>
#pragma omp parallel for schedule(static)
    2ab8:	b900bff3 	str	w19, [sp, #188]
    2abc:	b900bbf4 	str	w20, [sp, #184]
    2ac0:	f9005bf5 	str	x21, [sp, #176]
    2ac4:	f90057f9 	str	x25, [sp, #168]
    2ac8:	52800003 	mov	w3, #0x0                   	// #0
    2acc:	52800002 	mov	w2, #0x0                   	// #0
    2ad0:	aa1a03e1 	mov	x1, x26
    2ad4:	aa1803e0 	mov	x0, x24
    2ad8:	97fffa3e 	bl	13d0 <GOMP_parallel@plt>
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2adc:	11440294 	add	w20, w20, #0x100, lsl #12
    2ae0:	6b1402ff 	cmp	w23, w20
    2ae4:	54fffda8 	b.hi	2a98 <sptLoadSparseTensor+0x388>  // b.pmore
    2ae8:	a94573fb 	ldp	x27, x28, [sp, #80]
		free(ubuf);
    2aec:	aa1503e0 	mov	x0, x21
    2af0:	97fffa18 	bl	1350 <free@plt>
    2af4:	14000054 	b	2c44 <sptLoadSparseTensor+0x534>
	spt_CheckOSError(iores < 0, "SpTns Load");
    2af8:	97fffa32 	bl	13c0 <__errno_location@plt>
    2afc:	aa0003f4 	mov	x20, x0
    2b00:	b9400015 	ldr	w21, [x0]
    2b04:	314042bf 	cmn	w21, #0x10, lsl #12
    2b08:	54ffe5a0 	b.eq	27bc <sptLoadSparseTensor+0xac>  // b.none
    2b0c:	2a1503e0 	mov	w0, w21
    2b10:	97fff9ec 	bl	12c0 <strerror@plt>
    2b14:	aa0003e4 	mov	x4, x0
    2b18:	52800943 	mov	w3, #0x4a                  	// #74
    2b1c:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2b20:	9136a042 	add	x2, x2, #0xda8
    2b24:	114042a1 	add	w1, w21, #0x10, lsl #12
    2b28:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2b2c:	91374000 	add	x0, x0, #0xdd0
    2b30:	9400068f 	bl	456c <spt_ComplainError>
    2b34:	b9400280 	ldr	w0, [x20]
    2b38:	11404000 	add	w0, w0, #0x10, lsl #12
    2b3c:	14000041 	b	2c40 <sptLoadSparseTensor+0x530>
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    2b40:	97fffa20 	bl	13c0 <__errno_location@plt>
    2b44:	aa0003f9 	mov	x25, x0
    2b48:	b940001a 	ldr	w26, [x0]
    2b4c:	3140435f 	cmn	w26, #0x10, lsl #12
    2b50:	54ffe440 	b.eq	27d8 <sptLoadSparseTensor+0xc8>  // b.none
    2b54:	2a1a03e0 	mov	w0, w26
    2b58:	97fff9da 	bl	12c0 <strerror@plt>
    2b5c:	aa0003e4 	mov	x4, x0
    2b60:	528009a3 	mov	w3, #0x4d                  	// #77
    2b64:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2b68:	9136a042 	add	x2, x2, #0xda8
    2b6c:	11404341 	add	w1, w26, #0x10, lsl #12
    2b70:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2b74:	91374000 	add	x0, x0, #0xdd0
    2b78:	9400067d 	bl	456c <spt_ComplainError>
    2b7c:	b9400320 	ldr	w0, [x25]
    2b80:	11404000 	add	w0, w0, #0x10, lsl #12
    2b84:	1400002f 	b	2c40 <sptLoadSparseTensor+0x530>
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    2b88:	97fffa0e 	bl	13c0 <__errno_location@plt>
    2b8c:	aa0003f4 	mov	x20, x0
    2b90:	b9400018 	ldr	w24, [x0]
    2b94:	3140431f 	cmn	w24, #0x10, lsl #12
    2b98:	54ffe300 	b.eq	27f8 <sptLoadSparseTensor+0xe8>  // b.none
    2b9c:	2a1803e0 	mov	w0, w24
    2ba0:	97fff9c8 	bl	12c0 <strerror@plt>
    2ba4:	aa0003e4 	mov	x4, x0
    2ba8:	52800a03 	mov	w3, #0x50                  	// #80
    2bac:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2bb0:	9136a042 	add	x2, x2, #0xda8
    2bb4:	11404301 	add	w1, w24, #0x10, lsl #12
    2bb8:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2bbc:	91374000 	add	x0, x0, #0xdd0
    2bc0:	9400066b 	bl	456c <spt_ComplainError>
    2bc4:	b9400280 	ldr	w0, [x20]
    2bc8:	11404000 	add	w0, w0, #0x10, lsl #12
    2bcc:	1400001d 	b	2c40 <sptLoadSparseTensor+0x530>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2bd0:	11000694 	add	w20, w20, #0x1
    2bd4:	b9400275 	ldr	w21, [x19]
    2bd8:	6b15029f 	cmp	w20, w21
    2bdc:	54ffe162 	b.cs	2808 <sptLoadSparseTensor+0xf8>  // b.hs, b.nlast
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    2be0:	f9400a62 	ldr	x2, [x19, #16]
    2be4:	8b344842 	add	x2, x2, w20, uxtw #2
    2be8:	aa1803e1 	mov	x1, x24
    2bec:	aa1603e0 	mov	x0, x22
    2bf0:	97fff990 	bl	1230 <__isoc99_fscanf@plt>
		spt_CheckOSError(iores != 1, "SpTns Load");
    2bf4:	7100041f 	cmp	w0, #0x1
    2bf8:	54fffec0 	b.eq	2bd0 <sptLoadSparseTensor+0x4c0>  // b.none
    2bfc:	97fff9f1 	bl	13c0 <__errno_location@plt>
    2c00:	aa0003f5 	mov	x21, x0
    2c04:	b9400019 	ldr	w25, [x0]
    2c08:	3140433f 	cmn	w25, #0x10, lsl #12
    2c0c:	54fffe20 	b.eq	2bd0 <sptLoadSparseTensor+0x4c0>  // b.none
    2c10:	2a1903e0 	mov	w0, w25
    2c14:	97fff9ab 	bl	12c0 <strerror@plt>
    2c18:	aa0003e4 	mov	x4, x0
    2c1c:	52800a63 	mov	w3, #0x53                  	// #83
    2c20:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2c24:	9136a042 	add	x2, x2, #0xda8
    2c28:	11404321 	add	w1, w25, #0x10, lsl #12
    2c2c:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2c30:	91374000 	add	x0, x0, #0xdd0
    2c34:	9400064e 	bl	456c <spt_ComplainError>
    2c38:	b94002a0 	ldr	w0, [x21]
    2c3c:	11404000 	add	w0, w0, #0x10, lsl #12
		case 0:
			iores = p_tt_read_file(tsr, start_index, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
    2c40:	35000480 	cbnz	w0, 2cd0 <sptLoadSparseTensor+0x5c0>
			iores = p_tt_read_binary_file(tsr, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
			break;
	}

	fclose(fp);
    2c44:	aa1603e0 	mov	x0, x22
    2c48:	97fff96a 	bl	11f0 <fclose@plt>

	return 0;
    2c4c:	52800000 	mov	w0, #0x0                   	// #0
}
    2c50:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2c54:	a94153f3 	ldp	x19, x20, [sp, #16]
    2c58:	a94363f7 	ldp	x23, x24, [sp, #48]
    2c5c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2c60:	a8cc7bfd 	ldp	x29, x30, [sp], #192
    2c64:	d65f03c0 	ret
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2c68:	97fff9d6 	bl	13c0 <__errno_location@plt>
    2c6c:	aa0003f4 	mov	x20, x0
    2c70:	b9400015 	ldr	w21, [x0]
    2c74:	314042bf 	cmn	w21, #0x10, lsl #12
    2c78:	54ffdd60 	b.eq	2824 <sptLoadSparseTensor+0x114>  // b.none
    2c7c:	2a1503e0 	mov	w0, w21
    2c80:	97fff990 	bl	12c0 <strerror@plt>
    2c84:	aa0003e4 	mov	x4, x0
    2c88:	52800ae3 	mov	w3, #0x57                  	// #87
    2c8c:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2c90:	9136a042 	add	x2, x2, #0xda8
    2c94:	114042a1 	add	w1, w21, #0x10, lsl #12
    2c98:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2c9c:	91374000 	add	x0, x0, #0xdd0
    2ca0:	94000633 	bl	456c <spt_ComplainError>
    2ca4:	b9400280 	ldr	w0, [x20]
    2ca8:	11404000 	add	w0, w0, #0x10, lsl #12
    2cac:	17ffffe5 	b	2c40 <sptLoadSparseTensor+0x530>
		spt_CheckError(retval, "SpTns Load", NULL);
    2cb0:	d2800004 	mov	x4, #0x0                   	// #0
    2cb4:	52800b43 	mov	w3, #0x5a                  	// #90
    2cb8:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2cbc:	9136a042 	add	x2, x2, #0xda8
    2cc0:	2a0003e1 	mov	w1, w0
    2cc4:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2cc8:	91374000 	add	x0, x0, #0xdd0
    2ccc:	94000628 	bl	456c <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2cd0:	97fff9bc 	bl	13c0 <__errno_location@plt>
    2cd4:	aa0003f3 	mov	x19, x0
    2cd8:	b9400014 	ldr	w20, [x0]
    2cdc:	3140429f 	cmn	w20, #0x10, lsl #12
    2ce0:	54fffb20 	b.eq	2c44 <sptLoadSparseTensor+0x534>  // b.none
    2ce4:	2a1403e0 	mov	w0, w20
    2ce8:	97fff976 	bl	12c0 <strerror@plt>
    2cec:	aa0003e4 	mov	x4, x0
    2cf0:	52802483 	mov	w3, #0x124                 	// #292
    2cf4:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2cf8:	9136a042 	add	x2, x2, #0xda8
    2cfc:	11404281 	add	w1, w20, #0x10, lsl #12
    2d00:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2d04:	91374000 	add	x0, x0, #0xdd0
    2d08:	94000619 	bl	456c <spt_ComplainError>
    2d0c:	b9400260 	ldr	w0, [x19]
    2d10:	11404000 	add	w0, w0, #0x10, lsl #12
    2d14:	17ffffcf 	b	2c50 <sptLoadSparseTensor+0x540>
	spt_CheckError(retval, "SpTns Load", NULL);
    2d18:	d2800004 	mov	x4, #0x0                   	// #0
    2d1c:	52800ba3 	mov	w3, #0x5d                  	// #93
    2d20:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2d24:	9136a042 	add	x2, x2, #0xda8
    2d28:	2a0003e1 	mov	w1, w0
    2d2c:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2d30:	91374000 	add	x0, x0, #0xdd0
    2d34:	9400060e 	bl	456c <spt_ComplainError>
    2d38:	17ffffe6 	b	2cd0 <sptLoadSparseTensor+0x5c0>
				spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Load", "index < start_index");
    2d3c:	d0000004 	adrp	x4, 4000 <sptOmpMTTKRP+0x13c>
    2d40:	91378084 	add	x4, x4, #0xde0
    2d44:	52800d03 	mov	w3, #0x68                  	// #104
    2d48:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2d4c:	9136a042 	add	x2, x2, #0xda8
    2d50:	52800061 	mov	w1, #0x3                   	// #3
    2d54:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2d58:	91374000 	add	x0, x0, #0xdd0
    2d5c:	94000604 	bl	456c <spt_ComplainError>
    2d60:	17ffffdc 	b	2cd0 <sptLoadSparseTensor+0x5c0>
			iores = fscanf(fp, "%lf", &value);
    2d64:	aa1803e2 	mov	x2, x24
    2d68:	aa1a03e1 	mov	x1, x26
    2d6c:	aa1603e0 	mov	x0, x22
    2d70:	97fff930 	bl	1230 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2d74:	7100041f 	cmp	w0, #0x1
    2d78:	540003e1 	b.ne	2df4 <sptLoadSparseTensor+0x6e4>  // b.any
			sptAppendValueVector(&tsr->values, value);
    2d7c:	fd403fe0 	ldr	d0, [sp, #120]
    2d80:	1e624000 	fcvt	s0, d0
    2d84:	aa1903e0 	mov	x0, x25
    2d88:	97fffca7 	bl	2024 <sptAppendValueVector>
			++tsr->nnz;
    2d8c:	f9400e60 	ldr	x0, [x19, #24]
    2d90:	91000400 	add	x0, x0, #0x1
    2d94:	f9000e60 	str	x0, [x19, #24]
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2d98:	b9400260 	ldr	w0, [x19]
    2d9c:	52800014 	mov	w20, #0x0                   	// #0
    2da0:	34fffe20 	cbz	w0, 2d64 <sptLoadSparseTensor+0x654>
			iores = fscanf(fp, "%u", &index);
    2da4:	aa1803e2 	mov	x2, x24
    2da8:	aa1503e1 	mov	x1, x21
    2dac:	aa1603e0 	mov	x0, x22
    2db0:	97fff920 	bl	1230 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2db4:	7100041f 	cmp	w0, #0x1
    2db8:	540001e1 	b.ne	2df4 <sptLoadSparseTensor+0x6e4>  // b.any
			if(index < start_index) {
    2dbc:	b9407be1 	ldr	w1, [sp, #120]
    2dc0:	6b0102ff 	cmp	w23, w1
    2dc4:	54fffbc8 	b.hi	2d3c <sptLoadSparseTensor+0x62c>  // b.pmore
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2dc8:	d37f7e80 	ubfiz	x0, x20, #1, #32
    2dcc:	8b344000 	add	x0, x0, w20, uxtw
    2dd0:	f9401262 	ldr	x2, [x19, #32]
    2dd4:	4b170021 	sub	w1, w1, w23
    2dd8:	8b000c40 	add	x0, x2, x0, lsl #3
    2ddc:	97fffd30 	bl	229c <sptAppendIndexVector>
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2de0:	11000694 	add	w20, w20, #0x1
    2de4:	b9400260 	ldr	w0, [x19]
    2de8:	6b00029f 	cmp	w20, w0
    2dec:	54fffdc3 	b.cc	2da4 <sptLoadSparseTensor+0x694>  // b.lo, b.ul, b.last
    2df0:	17ffffdd 	b	2d64 <sptLoadSparseTensor+0x654>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2df4:	b9400261 	ldr	w1, [x19]
    2df8:	52800000 	mov	w0, #0x0                   	// #0
    2dfc:	34fff241 	cbz	w1, 2c44 <sptLoadSparseTensor+0x534>
		tsr->inds[mode].len = tsr->nnz;
    2e00:	f9400e63 	ldr	x3, [x19, #24]
    2e04:	f9401262 	ldr	x2, [x19, #32]
    2e08:	d37f7c01 	ubfiz	x1, x0, #1, #32
    2e0c:	8b204021 	add	x1, x1, w0, uxtw
    2e10:	f8217843 	str	x3, [x2, x1, lsl #3]
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e14:	11000400 	add	w0, w0, #0x1
    2e18:	b9400261 	ldr	w1, [x19]
    2e1c:	6b01001f 	cmp	w0, w1
    2e20:	54ffff03 	b.cc	2e00 <sptLoadSparseTensor+0x6f0>  // b.lo, b.ul, b.last
    2e24:	17ffff88 	b	2c44 <sptLoadSparseTensor+0x534>
    2e28:	a90573fb 	stp	x27, x28, [sp, #80]
		fprintf(stderr, "SPLATT: ERROR input has %lu-bit integers. "
    2e2c:	d37df002 	lsl	x2, x0, #3
    2e30:	aa0203e3 	mov	x3, x2
    2e34:	d0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    2e38:	91380021 	add	x1, x1, #0xe00
    2e3c:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    2e40:	f947d800 	ldr	x0, [x0, #4016]
    2e44:	f9400000 	ldr	x0, [x0]
    2e48:	97fff96a 	bl	13f0 <fprintf@plt>
		exit(-1);
    2e4c:	12800000 	mov	w0, #0xffffffff            	// #-1
    2e50:	97fff8cc 	bl	1180 <exit@plt>
		fprintf(stderr, "SPLATT: WARNING input has %lu-bit floating-point values. "
    2e54:	d37df002 	lsl	x2, x0, #3
    2e58:	aa0203e3 	mov	x3, x2
    2e5c:	d0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    2e60:	91394021 	add	x1, x1, #0xe50
    2e64:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    2e68:	f947d800 	ldr	x0, [x0, #4016]
    2e6c:	f9400000 	ldr	x0, [x0]
    2e70:	97fff960 	bl	13f0 <fprintf@plt>
    2e74:	17fffea7 	b	2910 <sptLoadSparseTensor+0x200>
		fread(buffer, sizeof(sptNnzIndex), count, fin);
    2e78:	aa1603e3 	mov	x3, x22
    2e7c:	d2800022 	mov	x2, #0x1                   	// #1
    2e80:	d2800101 	mov	x1, #0x8                   	// #8
    2e84:	9101c3e0 	add	x0, sp, #0x70
    2e88:	97fff92e 	bl	1340 <fread@plt>
    2e8c:	17fffec9 	b	29b0 <sptLoadSparseTensor+0x2a0>
		spt_CheckError(result, "SpTns Read", NULL);
    2e90:	d2800004 	mov	x4, #0x0                   	// #0
    2e94:	528020a3 	mov	w3, #0x105                 	// #261
    2e98:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2e9c:	9136a042 	add	x2, x2, #0xda8
    2ea0:	2a0003e1 	mov	w1, w0
    2ea4:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2ea8:	913b2000 	add	x0, x0, #0xec8
    2eac:	940005b0 	bl	456c <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2eb0:	97fff944 	bl	13c0 <__errno_location@plt>
    2eb4:	aa0003f3 	mov	x19, x0
    2eb8:	b9400014 	ldr	w20, [x0]
    2ebc:	3140429f 	cmn	w20, #0x10, lsl #12
    2ec0:	54ffec20 	b.eq	2c44 <sptLoadSparseTensor+0x534>  // b.none
    2ec4:	2a1403e0 	mov	w0, w20
    2ec8:	97fff8fe 	bl	12c0 <strerror@plt>
    2ecc:	aa0003e4 	mov	x4, x0
    2ed0:	52802503 	mov	w3, #0x128                 	// #296
    2ed4:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2ed8:	9136a042 	add	x2, x2, #0xda8
    2edc:	11404281 	add	w1, w20, #0x10, lsl #12
    2ee0:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2ee4:	91374000 	add	x0, x0, #0xdd0
    2ee8:	940005a1 	bl	456c <spt_ComplainError>
    2eec:	b9400260 	ldr	w0, [x19]
    2ef0:	11404000 	add	w0, w0, #0x10, lsl #12
    2ef4:	17ffff57 	b	2c50 <sptLoadSparseTensor+0x540>
	spt_CheckError(result, "SpTns Read", NULL);
    2ef8:	d2800004 	mov	x4, #0x0                   	// #0
    2efc:	52802103 	mov	w3, #0x108                 	// #264
    2f00:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2f04:	9136a042 	add	x2, x2, #0xda8
    2f08:	2a0003e1 	mov	w1, w0
    2f0c:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2f10:	913b2000 	add	x0, x0, #0xec8
    2f14:	94000596 	bl	456c <spt_ComplainError>
    2f18:	17ffffe6 	b	2eb0 <sptLoadSparseTensor+0x7a0>
		fread(buffer, sizeof(sptValue), count, fin);
    2f1c:	aa1603e3 	mov	x3, x22
    2f20:	2a1303e2 	mov	w2, w19
    2f24:	d2800081 	mov	x1, #0x4                   	// #4
    2f28:	aa1903e0 	mov	x0, x25
    2f2c:	97fff905 	bl	1340 <fread@plt>
    2f30:	17ffff45 	b	2c44 <sptLoadSparseTensor+0x534>

0000000000002f34 <sptNewMatrix>:
 * @param ncols the number of columns
 *
 * The memory layout of this dense matrix is a flat 2D array, with `ncols`
 * rounded up to multiples of 8
 */
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    2f34:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2f38:	910003fd 	mov	x29, sp
    2f3c:	a90153f3 	stp	x19, x20, [sp, #16]
    2f40:	a9025bf5 	stp	x21, x22, [sp, #32]
    2f44:	aa0003f4 	mov	x20, x0
	mtx->nrows = nrows;
    2f48:	b9000001 	str	w1, [x0]
	mtx->ncols = ncols;
    2f4c:	b9000402 	str	w2, [x0, #4]
	mtx->cap = nrows != 0 ? nrows : 1;
    2f50:	7100003f 	cmp	w1, #0x0
    2f54:	1a9f1433 	csinc	w19, w1, wzr, ne  // ne = any
    2f58:	b9000813 	str	w19, [x0, #8]
	mtx->stride = ((ncols-1)/8+1)*8;
    2f5c:	51000441 	sub	w1, w2, #0x1
    2f60:	121d7021 	and	w1, w1, #0xfffffff8
    2f64:	11002021 	add	w1, w1, #0x8
    2f68:	b9000c01 	str	w1, [x0, #12]
#ifdef _ISOC11_SOURCE
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    2f6c:	1b137c33 	mul	w19, w1, w19
    2f70:	2a1303f3 	mov	w19, w19
    2f74:	d37ef673 	lsl	x19, x19, #2
    2f78:	aa1303e1 	mov	x1, x19
    2f7c:	d2800400 	mov	x0, #0x20                  	// #32
    2f80:	97fff8b8 	bl	1260 <aligned_alloc@plt>
    2f84:	aa0003f5 	mov	x21, x0
    2f88:	f9000a80 	str	x0, [x20, #16]
		}
	}
#else
	mtx->values = malloc(mtx->cap * mtx->stride * sizeof (sptValue));
#endif
	spt_CheckOSError(!mtx->values, "Mtx New");
    2f8c:	b4000140 	cbz	x0, 2fb4 <sptNewMatrix+0x80>
	memset(mtx->values, 0, mtx->cap * mtx->stride * sizeof (sptValue));
    2f90:	aa1303e2 	mov	x2, x19
    2f94:	52800001 	mov	w1, #0x0                   	// #0
    2f98:	aa1503e0 	mov	x0, x21
    2f9c:	97fff8b5 	bl	1270 <memset@plt>
	return 0;
    2fa0:	52800000 	mov	w0, #0x0                   	// #0
}
    2fa4:	a94153f3 	ldp	x19, x20, [sp, #16]
    2fa8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2fac:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2fb0:	d65f03c0 	ret
	spt_CheckOSError(!mtx->values, "Mtx New");
    2fb4:	97fff903 	bl	13c0 <__errno_location@plt>
    2fb8:	aa0003f4 	mov	x20, x0
    2fbc:	b9400016 	ldr	w22, [x0]
    2fc0:	314042df 	cmn	w22, #0x10, lsl #12
    2fc4:	54fffe60 	b.eq	2f90 <sptNewMatrix+0x5c>  // b.none
    2fc8:	2a1603e0 	mov	w0, w22
    2fcc:	97fff8bd 	bl	12c0 <strerror@plt>
    2fd0:	aa0003e4 	mov	x4, x0
    2fd4:	52800703 	mov	w3, #0x38                  	// #56
    2fd8:	d0000002 	adrp	x2, 4000 <sptOmpMTTKRP+0x13c>
    2fdc:	913bc042 	add	x2, x2, #0xef0
    2fe0:	114042c1 	add	w1, w22, #0x10, lsl #12
    2fe4:	d0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    2fe8:	913c6000 	add	x0, x0, #0xf18
    2fec:	94000560 	bl	456c <spt_ComplainError>
    2ff0:	b9400280 	ldr	w0, [x20]
    2ff4:	11404000 	add	w0, w0, #0x10, lsl #12
    2ff8:	17ffffeb 	b	2fa4 <sptNewMatrix+0x70>

0000000000002ffc <sptRandomizeMatrix>:
 * @param ncols fill the specified number of columns
 *
 * The matrix is filled with uniform distributed pseudorandom number in [0, 1]
 * The random number will have a precision of 31 bits out of 51 bits
 */
int sptRandomizeMatrix(sptMatrix *mtx, bool random) {
    2ffc:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    3000:	910003fd 	mov	x29, sp
    3004:	a90153f3 	stp	x19, x20, [sp, #16]
    3008:	a9025bf5 	stp	x21, x22, [sp, #32]
    300c:	a90363f7 	stp	x23, x24, [sp, #48]
    3010:	a9046bf9 	stp	x25, x26, [sp, #64]
    3014:	aa0003f4 	mov	x20, x0
    3018:	12001c37 	and	w23, w1, #0xff
	for(sptIndex i=0; i<mtx->nrows; ++i)
    301c:	b9400000 	ldr	w0, [x0]
    3020:	52800015 	mov	w21, #0x0                   	// #0
		for(sptIndex j=0; j<mtx->ncols; ++j) {
			if (random) {
				srand(time(NULL) + (rand()%100) + i + j);
    3024:	5290a3fa 	mov	w26, #0x851f                	// #34079
    3028:	72aa3d7a 	movk	w26, #0x51eb, lsl #16
    302c:	52800c99 	mov	w25, #0x64                  	// #100
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3030:	340004c0 	cbz	w0, 30c8 <sptRandomizeMatrix+0xcc>
    3034:	f9002bfb 	str	x27, [sp, #80]
    3038:	1400001e 	b	30b0 <sptRandomizeMatrix+0xb4>
			} else {
				srand(1234 + i + j);
    303c:	0b130300 	add	w0, w24, w19
    3040:	97fff8d4 	bl	1390 <srand@plt>
			}
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3044:	b9400e96 	ldr	w22, [x20, #12]
    3048:	1b164eb6 	madd	w22, w21, w22, w19
    304c:	f9400a9b 	ldr	x27, [x20, #16]
    3050:	940005b8 	bl	4730 <sptRandomValue>
    3054:	bc367b60 	str	s0, [x27, x22, lsl #2]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3058:	11000673 	add	w19, w19, #0x1
    305c:	b9400680 	ldr	w0, [x20, #4]
    3060:	6b13001f 	cmp	w0, w19
    3064:	540001e9 	b.ls	30a0 <sptRandomizeMatrix+0xa4>  // b.plast
    3068:	0b1302b6 	add	w22, w21, w19
			if (random) {
    306c:	34fffe97 	cbz	w23, 303c <sptRandomizeMatrix+0x40>
				srand(time(NULL) + (rand()%100) + i + j);
    3070:	d2800000 	mov	x0, #0x0                   	// #0
    3074:	97fff867 	bl	1210 <time@plt>
    3078:	aa0003fb 	mov	x27, x0
    307c:	97fff885 	bl	1290 <rand@plt>
    3080:	9b3a7c01 	smull	x1, w0, w26
    3084:	9365fc21 	asr	x1, x1, #37
    3088:	4b807c21 	sub	w1, w1, w0, asr #31
    308c:	1b198021 	msub	w1, w1, w25, w0
    3090:	0b1b0020 	add	w0, w1, w27
    3094:	0b160000 	add	w0, w0, w22
    3098:	97fff8be 	bl	1390 <srand@plt>
    309c:	17ffffea 	b	3044 <sptRandomizeMatrix+0x48>
	for(sptIndex i=0; i<mtx->nrows; ++i)
    30a0:	110006b5 	add	w21, w21, #0x1
    30a4:	b9400280 	ldr	w0, [x20]
    30a8:	6b15001f 	cmp	w0, w21
    30ac:	540000c9 	b.ls	30c4 <sptRandomizeMatrix+0xc8>  // b.plast
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    30b0:	b9400680 	ldr	w0, [x20, #4]
    30b4:	52800013 	mov	w19, #0x0                   	// #0
				srand(1234 + i + j);
    30b8:	11134ab8 	add	w24, w21, #0x4d2
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    30bc:	35fffd60 	cbnz	w0, 3068 <sptRandomizeMatrix+0x6c>
    30c0:	17fffff8 	b	30a0 <sptRandomizeMatrix+0xa4>
    30c4:	f9402bfb 	ldr	x27, [sp, #80]
		}
	return 0;
}
    30c8:	52800000 	mov	w0, #0x0                   	// #0
    30cc:	a94153f3 	ldp	x19, x20, [sp, #16]
    30d0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    30d4:	a94363f7 	ldp	x23, x24, [sp, #48]
    30d8:	a9446bf9 	ldp	x25, x26, [sp, #64]
    30dc:	a8c67bfd 	ldp	x29, x30, [sp], #96
    30e0:	d65f03c0 	ret

00000000000030e4 <sptConstantMatrix>:
 * @param mtx   a pointer to a valid matrix
 * @param val   a given value constant
 *
 */
int sptConstantMatrix(sptMatrix *mtx, sptValue const val) {
	for(sptIndex i=0; i<mtx->nrows; ++i)
    30e4:	b9400001 	ldr	w1, [x0]
    30e8:	52800004 	mov	w4, #0x0                   	// #0
    30ec:	350001e1 	cbnz	w1, 3128 <sptConstantMatrix+0x44>
		for(sptIndex j=0; j<mtx->ncols; ++j)
			mtx->values[i * mtx->stride + j] = val;
	return 0;
}
    30f0:	52800000 	mov	w0, #0x0                   	// #0
    30f4:	d65f03c0 	ret
			mtx->values[i * mtx->stride + j] = val;
    30f8:	b9400c02 	ldr	w2, [x0, #12]
    30fc:	1b020482 	madd	w2, w4, w2, w1
    3100:	f9400803 	ldr	x3, [x0, #16]
    3104:	bc227860 	str	s0, [x3, x2, lsl #2]
		for(sptIndex j=0; j<mtx->ncols; ++j)
    3108:	11000421 	add	w1, w1, #0x1
    310c:	b9400402 	ldr	w2, [x0, #4]
    3110:	6b01005f 	cmp	w2, w1
    3114:	54ffff28 	b.hi	30f8 <sptConstantMatrix+0x14>  // b.pmore
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3118:	11000484 	add	w4, w4, #0x1
    311c:	b9400001 	ldr	w1, [x0]
    3120:	6b04003f 	cmp	w1, w4
    3124:	54fffe69 	b.ls	30f0 <sptConstantMatrix+0xc>  // b.plast
		for(sptIndex j=0; j<mtx->ncols; ++j)
    3128:	b9400402 	ldr	w2, [x0, #4]
    312c:	52800001 	mov	w1, #0x0                   	// #0
    3130:	35fffe42 	cbnz	w2, 30f8 <sptConstantMatrix+0x14>
    3134:	17fffff9 	b	3118 <sptConstantMatrix+0x34>

0000000000003138 <sptFreeMatrix>:
 * @param mtx a pointer to a valid matrix
 *
 * By using `sptFreeMatrix`, a valid matrix would become uninitialized and
 * should not be used anymore prior to another initialization
 */
void sptFreeMatrix(sptMatrix *mtx) {
    3138:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    313c:	910003fd 	mov	x29, sp
    3140:	f9000bf3 	str	x19, [sp, #16]
    3144:	aa0003f3 	mov	x19, x0
	free(mtx->values);
    3148:	f9400800 	ldr	x0, [x0, #16]
    314c:	97fff881 	bl	1350 <free@plt>
	mtx->nrows = 0;
    3150:	b900027f 	str	wzr, [x19]
	mtx->ncols = 0;
    3154:	b900067f 	str	wzr, [x19, #4]
	mtx->cap = 0;
    3158:	b9000a7f 	str	wzr, [x19, #8]
	mtx->stride = 0;
    315c:	b9000e7f 	str	wzr, [x19, #12]
}
    3160:	f9400bf3 	ldr	x19, [sp, #16]
    3164:	a8c27bfd 	ldp	x29, x30, [sp], #32
    3168:	d65f03c0 	ret

000000000000316c <sptSparseTensorDensity>:
#include <bits/types/FILE.h>
#include <stdio.h>


double sptSparseTensorDensity(sptSparseTensor const * const tsr)
{
    316c:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    3170:	910003fd 	mov	x29, sp
    3174:	f9000bf3 	str	x19, [sp, #16]
    3178:	aa0003f3 	mov	x19, x0
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    317c:	b9400000 	ldr	w0, [x0]
    3180:	1e630001 	ucvtf	d1, w0
    3184:	fd400e60 	ldr	d0, [x19, #24]
    3188:	1e6e1002 	fmov	d2, #1.000000000000000000e+00
    318c:	1e611841 	fdiv	d1, d2, d1
    3190:	7e61d800 	ucvtf	d0, d0
    3194:	97fff80f 	bl	11d0 <pow@plt>
	double density = 1.0;
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3198:	b9400262 	ldr	w2, [x19]
    319c:	340001e2 	cbz	w2, 31d8 <sptSparseTensorDensity+0x6c>
    31a0:	1e604002 	fmov	d2, d0
		density *= root / (double)tsr->ndims[m];
    31a4:	f9400a63 	ldr	x3, [x19, #16]
    31a8:	d2800000 	mov	x0, #0x0                   	// #0
	double density = 1.0;
    31ac:	1e6e1000 	fmov	d0, #1.000000000000000000e+00
		density *= root / (double)tsr->ndims[m];
    31b0:	b8607861 	ldr	w1, [x3, x0, lsl #2]
    31b4:	1e630021 	ucvtf	d1, w1
    31b8:	1e611841 	fdiv	d1, d2, d1
    31bc:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    31c0:	91000400 	add	x0, x0, #0x1
    31c4:	6b00005f 	cmp	w2, w0
    31c8:	54ffff48 	b.hi	31b0 <sptSparseTensorDensity+0x44>  // b.pmore
	}

	return density;
}
    31cc:	f9400bf3 	ldr	x19, [sp, #16]
    31d0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    31d4:	d65f03c0 	ret
	double density = 1.0;
    31d8:	1e6e1000 	fmov	d0, #1.000000000000000000e+00
	return density;
    31dc:	17fffffc 	b	31cc <sptSparseTensorDensity+0x60>

00000000000031e0 <sptSparseTensorStatus>:



void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp)
{
    31e0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    31e4:	910003fd 	mov	x29, sp
    31e8:	a90153f3 	stp	x19, x20, [sp, #16]
    31ec:	a9025bf5 	stp	x21, x22, [sp, #32]
    31f0:	aa0003f3 	mov	x19, x0
    31f4:	aa0103f5 	mov	x21, x1
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    31f8:	aa0103e3 	mov	x3, x1
    31fc:	d2800822 	mov	x2, #0x41                  	// #65
    3200:	d2800021 	mov	x1, #0x1                   	// #1
    3204:	b0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    3208:	913c8000 	add	x0, x0, #0xf20
    320c:	97fff855 	bl	1360 <fwrite@plt>
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
    3210:	f9400a60 	ldr	x0, [x19, #16]
    3214:	b9400002 	ldr	w2, [x0]
    3218:	b0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    321c:	913da021 	add	x1, x1, #0xf68
    3220:	aa1503e0 	mov	x0, x21
    3224:	97fff873 	bl	13f0 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3228:	b9400260 	ldr	w0, [x19]
    322c:	7100041f 	cmp	w0, #0x1
    3230:	540001a9 	b.ls	3264 <sptSparseTensorStatus+0x84>  // b.plast
    3234:	52800034 	mov	w20, #0x1                   	// #1
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    3238:	b0000016 	adrp	x22, 4000 <sptOmpMTTKRP+0x13c>
    323c:	913de2d6 	add	x22, x22, #0xf78
    3240:	f9400a60 	ldr	x0, [x19, #16]
    3244:	b8745802 	ldr	w2, [x0, w20, uxtw #2]
    3248:	aa1603e1 	mov	x1, x22
    324c:	aa1503e0 	mov	x0, x21
    3250:	97fff868 	bl	13f0 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3254:	11000694 	add	w20, w20, #0x1
    3258:	b9400260 	ldr	w0, [x19]
    325c:	6b14001f 	cmp	w0, w20
    3260:	54ffff08 	b.hi	3240 <sptSparseTensorStatus+0x60>  // b.pmore
	}
	fprintf(fp, " NNZ = %"PASTA_PRI_NNZ_INDEX, tsr->nnz);
    3264:	f9400e62 	ldr	x2, [x19, #24]
    3268:	b0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    326c:	913e0021 	add	x1, x1, #0xf80
    3270:	aa1503e0 	mov	x0, x21
    3274:	97fff85f 	bl	13f0 <fprintf@plt>
	fprintf(fp, " DENSITY = %e\n" , sptSparseTensorDensity(tsr));
    3278:	aa1303e0 	mov	x0, x19
    327c:	97ffffbc 	bl	316c <sptSparseTensorDensity>
    3280:	b0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    3284:	913e4021 	add	x1, x1, #0xf90
    3288:	aa1503e0 	mov	x0, x21
    328c:	97fff859 	bl	13f0 <fprintf@plt>

	fprintf(fp, "Average slice length (c): ");
    3290:	aa1503e3 	mov	x3, x21
    3294:	d2800342 	mov	x2, #0x1a                  	// #26
    3298:	d2800021 	mov	x1, #0x1                   	// #1
    329c:	b0000000 	adrp	x0, 4000 <sptOmpMTTKRP+0x13c>
    32a0:	913e8000 	add	x0, x0, #0xfa0
    32a4:	97fff82f 	bl	1360 <fwrite@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    32a8:	b9400260 	ldr	w0, [x19]
    32ac:	7100041f 	cmp	w0, #0x1
    32b0:	54000600 	b.eq	3370 <sptSparseTensorStatus+0x190>  // b.none
    32b4:	52800014 	mov	w20, #0x0                   	// #0
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    32b8:	b0000016 	adrp	x22, 4000 <sptOmpMTTKRP+0x13c>
    32bc:	913f02d6 	add	x22, x22, #0xfc0
    32c0:	fd400e60 	ldr	d0, [x19, #24]
    32c4:	7e61d800 	ucvtf	d0, d0
    32c8:	f9400a60 	ldr	x0, [x19, #16]
    32cc:	b8745800 	ldr	w0, [x0, w20, uxtw #2]
    32d0:	1e630001 	ucvtf	d1, w0
    32d4:	1e611800 	fdiv	d0, d0, d1
    32d8:	aa1603e1 	mov	x1, x22
    32dc:	aa1503e0 	mov	x0, x21
    32e0:	97fff844 	bl	13f0 <fprintf@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    32e4:	11000694 	add	w20, w20, #0x1
    32e8:	b9400262 	ldr	w2, [x19]
    32ec:	51000442 	sub	w2, w2, #0x1
    32f0:	6b14005f 	cmp	w2, w20
    32f4:	54fffe68 	b.hi	32c0 <sptSparseTensorStatus+0xe0>  // b.pmore
	}
	fprintf(fp, "%.2lf\n", (double)tsr->nnz / tsr->ndims[tsr->nmodes-1]);
    32f8:	fd400e60 	ldr	d0, [x19, #24]
    32fc:	7e61d800 	ucvtf	d0, d0
    3300:	f9400a60 	ldr	x0, [x19, #16]
    3304:	b8625800 	ldr	w0, [x0, w2, uxtw #2]
    3308:	1e630001 	ucvtf	d1, w0
    330c:	1e611800 	fdiv	d0, d0, d1
    3310:	b0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    3314:	913f4021 	add	x1, x1, #0xfd0
    3318:	aa1503e0 	mov	x0, x21
    331c:	97fff835 	bl	13f0 <fprintf@plt>

	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    3320:	b9400261 	ldr	w1, [x19]
    3324:	f9400e60 	ldr	x0, [x19, #24]
    3328:	9b000020 	madd	x0, x1, x0, x0
    332c:	d37ef400 	lsl	x0, x0, #2
    3330:	940004cc 	bl	4660 <sptBytesString>
    3334:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    3338:	aa0003e2 	mov	x2, x0
    333c:	b0000001 	adrp	x1, 4000 <sptOmpMTTKRP+0x13c>
    3340:	913f6021 	add	x1, x1, #0xfd8
    3344:	aa1503e0 	mov	x0, x21
    3348:	97fff82a 	bl	13f0 <fprintf@plt>
	fprintf(fp, "\n");
    334c:	aa1503e1 	mov	x1, x21
    3350:	52800140 	mov	w0, #0xa                   	// #10
    3354:	97fff793 	bl	11a0 <fputc@plt>
	free(bytestr);
    3358:	aa1303e0 	mov	x0, x19
    335c:	97fff7fd 	bl	1350 <free@plt>
}
    3360:	a94153f3 	ldp	x19, x20, [sp, #16]
    3364:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3368:	a8c37bfd 	ldp	x29, x30, [sp], #48
    336c:	d65f03c0 	ret
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    3370:	52800002 	mov	w2, #0x0                   	// #0
    3374:	17ffffe1 	b	32f8 <sptSparseTensorStatus+0x118>

0000000000003378 <sptMTTKRP_3D>:

int sptMTTKRP_3D(sptSparseTensor const * const X,
								 sptMatrix * mats[],     // mats[nmodes] as temporary space.
								 sptIndex const mats_order[],    // Correspond to the mode order of X.
								 sptIndex const mode)
{
    3378:	a9b87bfd 	stp	x29, x30, [sp, #-128]!
    337c:	910003fd 	mov	x29, sp
    3380:	a90573fb 	stp	x27, x28, [sp, #80]
    3384:	aa0003fb 	mov	x27, x0
	sptIndex const nmodes = X->nmodes;
	sptNnzIndex const nnz = X->nnz;
	sptIndex const * const ndims = X->ndims;
    3388:	f9400805 	ldr	x5, [x0, #16]
	sptValue const * const restrict vals = X->values.data;
	sptIndex const stride = mats[0]->stride;

	/* Check the mats. */
	sptAssert(nmodes ==3);
    338c:	b9400000 	ldr	w0, [x0]
    3390:	71000c1f 	cmp	w0, #0x3
    3394:	54000e01 	b.ne	3554 <sptMTTKRP_3D+0x1dc>  // b.any
    3398:	a9025bf5 	stp	x21, x22, [sp, #32]
    339c:	aa0103f5 	mov	x21, x1
    33a0:	aa0203fc 	mov	x28, x2
	for(sptIndex i=0; i<nmodes; ++i) {
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    33a4:	f9400c26 	ldr	x6, [x1, #24]
    33a8:	b94004c4 	ldr	w4, [x6, #4]
    33ac:	d2800000 	mov	x0, #0x0                   	// #0
    33b0:	f8607aa1 	ldr	x1, [x21, x0, lsl #3]
    33b4:	b9400422 	ldr	w2, [x1, #4]
    33b8:	6b04005f 	cmp	w2, w4
    33bc:	54000da1 	b.ne	3570 <sptMTTKRP_3D+0x1f8>  // b.any
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
		}
		if(mats[i]->nrows != ndims[i]) {
    33c0:	b9400022 	ldr	w2, [x1]
    33c4:	b86078a1 	ldr	w1, [x5, x0, lsl #2]
    33c8:	6b01005f 	cmp	w2, w1
    33cc:	54000e81 	b.ne	359c <sptMTTKRP_3D+0x224>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    33d0:	91000400 	add	x0, x0, #0x1
    33d4:	f1000c1f 	cmp	x0, #0x3
    33d8:	54fffec1 	b.ne	33b0 <sptMTTKRP_3D+0x38>  // b.any
    33dc:	a90153f3 	stp	x19, x20, [sp, #16]
    33e0:	a90363f7 	stp	x23, x24, [sp, #48]
    33e4:	a9046bf9 	stp	x25, x26, [sp, #64]
    33e8:	fd0033e8 	str	d8, [sp, #96]
	sptNnzIndex const nnz = X->nnz;
    33ec:	f9400f78 	ldr	x24, [x27, #24]
	sptValue const * const restrict vals = X->values.data;
    33f0:	f9401f79 	ldr	x25, [x27, #56]
	sptIndex const stride = mats[0]->stride;
    33f4:	f94002a0 	ldr	x0, [x21]
    33f8:	b9400c16 	ldr	w22, [x0, #12]
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
		}
	}


	sptIndex const tmpI = mats[mode]->nrows;
    33fc:	2a0303e0 	mov	w0, w3
    3400:	f8607aa1 	ldr	x1, [x21, x0, lsl #3]
	sptIndex const R = mats[mode]->ncols;
    3404:	b9400437 	ldr	w23, [x1, #4]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3408:	f9401362 	ldr	x2, [x27, #32]
    340c:	8b000400 	add	x0, x0, x0, lsl #1
    3410:	8b000c40 	add	x0, x2, x0, lsl #3
    3414:	f940081a 	ldr	x26, [x0, #16]
	sptMatrix * const restrict M = mats[nmodes];
	sptValue * const restrict mvals = M->values;
    3418:	f94008d3 	ldr	x19, [x6, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    341c:	b9400022 	ldr	w2, [x1]
    3420:	1b027ec2 	mul	w2, w22, w2
    3424:	d37ef442 	lsl	x2, x2, #2
    3428:	52800001 	mov	w1, #0x0                   	// #0
    342c:	aa1303e0 	mov	x0, x19
    3430:	97fff790 	bl	1270 <memset@plt>

	sptIndex times_mat_index_1 = mats_order[1];
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    3434:	b9400780 	ldr	w0, [x28, #4]
    3438:	f8607ab4 	ldr	x20, [x21, x0, lsl #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    343c:	f9401361 	ldr	x1, [x27, #32]
    3440:	8b000400 	add	x0, x0, x0, lsl #1
    3444:	8b000c20 	add	x0, x1, x0, lsl #3
    3448:	f940081b 	ldr	x27, [x0, #16]
	sptIndex times_mat_index_2 = mats_order[2];
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    344c:	b9400b80 	ldr	w0, [x28, #8]
    3450:	f8607ab5 	ldr	x21, [x21, x0, lsl #3]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    3454:	8b000400 	add	x0, x0, x0, lsl #1
    3458:	8b000c20 	add	x0, x1, x0, lsl #3
    345c:	f940081c 	ldr	x28, [x0, #16]
	sptIndex mode_i;
	sptIndex tmp_i_1, tmp_i_2;
	sptValue entry;

	sptTimer timer;
	sptNewTimer(&timer, 0);
    3460:	52800001 	mov	w1, #0x0                   	// #0
    3464:	9101e3e0 	add	x0, sp, #0x78
    3468:	94000315 	bl	40bc <sptNewTimer>
	double comp_time, total_time;

	sptStartTimer(timer);
    346c:	f9403fe0 	ldr	x0, [sp, #120]
    3470:	9400032d 	bl	4124 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3474:	b4000418 	cbz	x24, 34f4 <sptMTTKRP_3D+0x17c>
    3478:	d2800007 	mov	x7, #0x0                   	// #0
		mode_i = mode_ind[x];
    347c:	b8677b44 	ldr	w4, [x26, x7, lsl #2]
		tmp_i_1 = times_inds_1[x];
    3480:	b8677b65 	ldr	w5, [x27, x7, lsl #2]
		tmp_i_2 = times_inds_2[x];
    3484:	b8677b80 	ldr	w0, [x28, x7, lsl #2]
		entry = vals[x];
    3488:	bc677b22 	ldr	s2, [x25, x7, lsl #2]

		for(sptIndex r=0; r<R; ++r) {
    348c:	340002f7 	cbz	w23, 34e8 <sptMTTKRP_3D+0x170>
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3490:	1b047ec1 	mul	w1, w22, w4
    3494:	1b057ec5 	mul	w5, w22, w5
    3498:	1b007ec4 	mul	w4, w22, w0
    349c:	0b170026 	add	w6, w1, w23
    34a0:	2a0103e0 	mov	w0, w1
    34a4:	4b0100a5 	sub	w5, w5, w1
    34a8:	4b010084 	sub	w4, w4, w1
    34ac:	2a0003e1 	mov	w1, w0
    34b0:	0b0000a3 	add	w3, w5, w0
    34b4:	f9400a82 	ldr	x2, [x20, #16]
    34b8:	bc637840 	ldr	s0, [x2, x3, lsl #2]
    34bc:	1e200840 	fmul	s0, s2, s0
    34c0:	0b000083 	add	w3, w4, w0
    34c4:	f9400aa2 	ldr	x2, [x21, #16]
    34c8:	bc637841 	ldr	s1, [x2, x3, lsl #2]
    34cc:	1e210800 	fmul	s0, s0, s1
    34d0:	bc617a61 	ldr	s1, [x19, x1, lsl #2]
    34d4:	1e202820 	fadd	s0, s1, s0
    34d8:	bc217a60 	str	s0, [x19, x1, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    34dc:	11000400 	add	w0, w0, #0x1
    34e0:	6b06001f 	cmp	w0, w6
    34e4:	54fffe41 	b.ne	34ac <sptMTTKRP_3D+0x134>  // b.any
	for(sptNnzIndex x=0; x<nnz; ++x) {
    34e8:	910004e7 	add	x7, x7, #0x1
    34ec:	eb07031f 	cmp	x24, x7
    34f0:	54fffc61 	b.ne	347c <sptMTTKRP_3D+0x104>  // b.any
		}
	}
	sptStopTimer(timer);
    34f4:	f9403fe0 	ldr	x0, [sp, #120]
    34f8:	9400033b 	bl	41e4 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    34fc:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    3500:	91010021 	add	x1, x1, #0x40
    3504:	f9403fe0 	ldr	x0, [sp, #120]
    3508:	9400037b 	bl	42f4 <sptPrintElapsedTime>
    350c:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3510:	f9403fe0 	ldr	x0, [sp, #120]
    3514:	940003a1 	bl	4398 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    3518:	1e604100 	fmov	d0, d8
    351c:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    3520:	9101e000 	add	x0, x0, #0x78
    3524:	97fff7a3 	bl	13b0 <printf@plt>
	printf("\n");
    3528:	52800140 	mov	w0, #0xa                   	// #10
    352c:	97fff7ad 	bl	13e0 <putchar@plt>

	return 0;
    3530:	52800000 	mov	w0, #0x0                   	// #0
    3534:	a94153f3 	ldp	x19, x20, [sp, #16]
    3538:	a94363f7 	ldp	x23, x24, [sp, #48]
    353c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3540:	fd4033e8 	ldr	d8, [sp, #96]
}
    3544:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3548:	a94573fb 	ldp	x27, x28, [sp, #80]
    354c:	a8c87bfd 	ldp	x29, x30, [sp], #128
    3550:	d65f03c0 	ret
    3554:	a90153f3 	stp	x19, x20, [sp, #16]
    3558:	a9025bf5 	stp	x21, x22, [sp, #32]
    355c:	a90363f7 	stp	x23, x24, [sp, #48]
    3560:	a9046bf9 	stp	x25, x26, [sp, #64]
    3564:	fd0033e8 	str	d8, [sp, #96]
	sptAssert(nmodes ==3);
    3568:	12800000 	mov	w0, #0xffffffff            	// #-1
    356c:	97fff705 	bl	1180 <exit@plt>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3570:	b0000004 	adrp	x4, 4000 <sptOmpMTTKRP+0x13c>
    3574:	913fc084 	add	x4, x4, #0xff0
    3578:	52801263 	mov	w3, #0x93                  	// #147
    357c:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    3580:	91006042 	add	x2, x2, #0x18
    3584:	52800041 	mov	w1, #0x2                   	// #2
    3588:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    358c:	91010000 	add	x0, x0, #0x40
    3590:	940003f7 	bl	456c <spt_ComplainError>
    3594:	52800040 	mov	w0, #0x2                   	// #2
    3598:	17ffffeb 	b	3544 <sptMTTKRP_3D+0x1cc>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    359c:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7b8>
    35a0:	91016084 	add	x4, x4, #0x58
    35a4:	528012c3 	mov	w3, #0x96                  	// #150
    35a8:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    35ac:	91006042 	add	x2, x2, #0x18
    35b0:	52800041 	mov	w1, #0x2                   	// #2
    35b4:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    35b8:	91010000 	add	x0, x0, #0x40
    35bc:	940003ec 	bl	456c <spt_ComplainError>
    35c0:	52800040 	mov	w0, #0x2                   	// #2
    35c4:	17ffffe0 	b	3544 <sptMTTKRP_3D+0x1cc>

00000000000035c8 <sptMTTKRP>:
							sptIndex const mode) {
    35c8:	a9b67bfd 	stp	x29, x30, [sp, #-160]!
    35cc:	910003fd 	mov	x29, sp
    35d0:	a9025bf5 	stp	x21, x22, [sp, #32]
    35d4:	a90363f7 	stp	x23, x24, [sp, #48]
    35d8:	aa0003f7 	mov	x23, x0
    35dc:	aa0103f5 	mov	x21, x1
    35e0:	aa0203f8 	mov	x24, x2
	sptIndex const nmodes = X->nmodes;
    35e4:	b9400016 	ldr	w22, [x0]
	if(nmodes == 3) {
    35e8:	71000edf 	cmp	w22, #0x3
    35ec:	54001260 	b.eq	3838 <sptMTTKRP+0x270>  // b.none
	sptIndex const * const ndims = X->ndims;
    35f0:	f9400805 	ldr	x5, [x0, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    35f4:	340001f6 	cbz	w22, 3630 <sptMTTKRP+0x68>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    35f8:	f8765820 	ldr	x0, [x1, w22, uxtw #3]
    35fc:	b9400404 	ldr	w4, [x0, #4]
    3600:	d2800000 	mov	x0, #0x0                   	// #0
    3604:	f8607aa1 	ldr	x1, [x21, x0, lsl #3]
    3608:	b9400422 	ldr	w2, [x1, #4]
    360c:	6b04005f 	cmp	w2, w4
    3610:	54001241 	b.ne	3858 <sptMTTKRP+0x290>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    3614:	b9400022 	ldr	w2, [x1]
    3618:	b86078a1 	ldr	w1, [x5, x0, lsl #2]
    361c:	6b01005f 	cmp	w2, w1
    3620:	54001321 	b.ne	3884 <sptMTTKRP+0x2bc>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    3624:	91000400 	add	x0, x0, #0x1
    3628:	6b0002df 	cmp	w22, w0
    362c:	54fffec8 	b.hi	3604 <sptMTTKRP+0x3c>  // b.pmore
    3630:	a90153f3 	stp	x19, x20, [sp, #16]
    3634:	a9046bf9 	stp	x25, x26, [sp, #64]
    3638:	a90573fb 	stp	x27, x28, [sp, #80]
    363c:	fd0033e8 	str	d8, [sp, #96]
	sptNnzIndex const nnz = X->nnz;
    3640:	f9400efa 	ldr	x26, [x23, #24]
	sptValue const * const restrict vals = X->values.data;
    3644:	f9401efb 	ldr	x27, [x23, #56]
	sptIndex const stride = mats[0]->stride;
    3648:	f94002a0 	ldr	x0, [x21]
    364c:	b9400c19 	ldr	w25, [x0, #12]
	sptIndex const tmpI = mats[mode]->nrows;
    3650:	2a0303e3 	mov	w3, w3
    3654:	f8637aa1 	ldr	x1, [x21, x3, lsl #3]
	sptIndex const R = mats[mode]->ncols;
    3658:	b9400433 	ldr	w19, [x1, #4]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    365c:	f94012e0 	ldr	x0, [x23, #32]
    3660:	8b030463 	add	x3, x3, x3, lsl #1
    3664:	8b030c00 	add	x0, x0, x3, lsl #3
    3668:	f940081c 	ldr	x28, [x0, #16]
	sptValue * const restrict mvals = M->values;
    366c:	f8765aa0 	ldr	x0, [x21, w22, uxtw #3]
    3670:	f9400814 	ldr	x20, [x0, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3674:	b9400022 	ldr	w2, [x1]
    3678:	1b027f22 	mul	w2, w25, w2
    367c:	d37ef442 	lsl	x2, x2, #2
    3680:	52800001 	mov	w1, #0x0                   	// #0
    3684:	aa1403e0 	mov	x0, x20
    3688:	97fff6fa 	bl	1270 <memset@plt>
	sptNewValueVector(&scratch, R, R);
    368c:	2a1303e1 	mov	w1, w19
    3690:	910223e0 	add	x0, sp, #0x88
    3694:	aa0103e2 	mov	x2, x1
    3698:	f9003fe0 	str	x0, [sp, #120]
    369c:	97fffa2a 	bl	1f44 <sptNewValueVector>
	sptConstantValueVector(&scratch, 0);
    36a0:	0f000400 	movi	v0.2s, #0x0
    36a4:	f9403fe0 	ldr	x0, [sp, #120]
    36a8:	97fffa54 	bl	1ff8 <sptConstantValueVector>
	sptNewTimer(&timer, 0);
    36ac:	52800001 	mov	w1, #0x0                   	// #0
    36b0:	910203e0 	add	x0, sp, #0x80
    36b4:	94000282 	bl	40bc <sptNewTimer>
	sptStartTimer(timer);
    36b8:	f94043e0 	ldr	x0, [sp, #128]
    36bc:	9400029a 	bl	4124 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    36c0:	b400089a 	cbz	x26, 37d0 <sptMTTKRP+0x208>
    36c4:	d37ef75a 	lsl	x26, x26, #2
    36c8:	51000665 	sub	w5, w19, #0x1
    36cc:	910004a5 	add	x5, x5, #0x1
    36d0:	d37ef4a5 	lsl	x5, x5, #2
    36d4:	d2800004 	mov	x4, #0x0                   	// #0
		sptMatrix * times_mat = mats[times_mat_index];
    36d8:	b9400700 	ldr	w0, [x24, #4]
		sptValue const entry = vals[x];
    36dc:	bc646b61 	ldr	s1, [x27, x4]
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    36e0:	f8607aa1 	ldr	x1, [x21, x0, lsl #3]
    36e4:	f9400822 	ldr	x2, [x1, #16]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    36e8:	f94012e1 	ldr	x1, [x23, #32]
    36ec:	8b000400 	add	x0, x0, x0, lsl #1
    36f0:	8b000c20 	add	x0, x1, x0, lsl #3
		sptIndex tmp_i = times_inds[x];
    36f4:	f9400800 	ldr	x0, [x0, #16]
		sptIndex tmp_mult = tmp_i * stride;
    36f8:	b8646800 	ldr	w0, [x0, x4]
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    36fc:	1b007f20 	mul	w0, w25, w0
		for(sptIndex r=0; r<R; ++r) {
    3700:	34000153 	cbz	w19, 3728 <sptMTTKRP+0x160>
    3704:	8b000842 	add	x2, x2, x0, lsl #2
    3708:	d2800000 	mov	x0, #0x0                   	// #0
			scratch.data[r] = entry * times_mat_values[r];
    370c:	bc606840 	ldr	s0, [x2, x0]
    3710:	1e210800 	fmul	s0, s0, s1
    3714:	f9404fe1 	ldr	x1, [sp, #152]
    3718:	bc206820 	str	s0, [x1, x0]
		for(sptIndex r=0; r<R; ++r) {
    371c:	91001000 	add	x0, x0, #0x4
    3720:	eb05001f 	cmp	x0, x5
    3724:	54ffff41 	b.ne	370c <sptMTTKRP+0x144>  // b.any
		for(sptIndex i=2; i<nmodes; ++i) {
    3728:	71000adf 	cmp	w22, #0x2
    372c:	54000329 	b.ls	3790 <sptMTTKRP+0x1c8>  // b.plast
    3730:	d2800043 	mov	x3, #0x2                   	// #2
			times_mat = mats[times_mat_index];
    3734:	b8637b00 	ldr	w0, [x24, x3, lsl #2]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3738:	f8607aa1 	ldr	x1, [x21, x0, lsl #3]
    373c:	f9400822 	ldr	x2, [x1, #16]
			times_inds = X->inds[times_mat_index].data;
    3740:	f94012e1 	ldr	x1, [x23, #32]
    3744:	8b000400 	add	x0, x0, x0, lsl #1
    3748:	8b000c20 	add	x0, x1, x0, lsl #3
			tmp_i = times_inds[x];
    374c:	f9400800 	ldr	x0, [x0, #16]
			tmp_mult = tmp_i * stride;
    3750:	b8646800 	ldr	w0, [x0, x4]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3754:	1b007f20 	mul	w0, w25, w0
			sptValue* scratch_data = scratch.data;
    3758:	f9404fe1 	ldr	x1, [sp, #152]
			for(sptIndex r=0; r<R; ++r) {
    375c:	34000153 	cbz	w19, 3784 <sptMTTKRP+0x1bc>
    3760:	8b000842 	add	x2, x2, x0, lsl #2
    3764:	d2800000 	mov	x0, #0x0                   	// #0
				scratch_data[r] *= times_mat_values[r];
    3768:	bc607820 	ldr	s0, [x1, x0, lsl #2]
    376c:	bc607841 	ldr	s1, [x2, x0, lsl #2]
    3770:	1e210800 	fmul	s0, s0, s1
    3774:	bc207820 	str	s0, [x1, x0, lsl #2]
			for(sptIndex r=0; r<R; ++r) {
    3778:	91000400 	add	x0, x0, #0x1
    377c:	6b00027f 	cmp	w19, w0
    3780:	54ffff48 	b.hi	3768 <sptMTTKRP+0x1a0>  // b.pmore
		for(sptIndex i=2; i<nmodes; ++i) {
    3784:	91000463 	add	x3, x3, #0x1
    3788:	6b0302df 	cmp	w22, w3
    378c:	54fffd48 	b.hi	3734 <sptMTTKRP+0x16c>  // b.pmore
		sptIndex tmp_mode = mode_i * stride;
    3790:	b8646b82 	ldr	w2, [x28, x4]
    3794:	1b027f22 	mul	w2, w25, w2
		sptValue* scratch_data = scratch.data;
    3798:	f9404fe3 	ldr	x3, [sp, #152]
		for(sptIndex r=0; r<R; ++r) {
    379c:	34000153 	cbz	w19, 37c4 <sptMTTKRP+0x1fc>
    37a0:	d2800000 	mov	x0, #0x0                   	// #0
			mvals[tmp_mode + r] += scratch_data[r];
    37a4:	0b000041 	add	w1, w2, w0
    37a8:	bc617a80 	ldr	s0, [x20, x1, lsl #2]
    37ac:	bc607861 	ldr	s1, [x3, x0, lsl #2]
    37b0:	1e212800 	fadd	s0, s0, s1
    37b4:	bc217a80 	str	s0, [x20, x1, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    37b8:	91000400 	add	x0, x0, #0x1
    37bc:	6b00027f 	cmp	w19, w0
    37c0:	54ffff28 	b.hi	37a4 <sptMTTKRP+0x1dc>  // b.pmore
	for(sptNnzIndex x=0; x<nnz; ++x) {
    37c4:	91001084 	add	x4, x4, #0x4
    37c8:	eb1a009f 	cmp	x4, x26
    37cc:	54fff861 	b.ne	36d8 <sptMTTKRP+0x110>  // b.any
	sptStopTimer(timer);
    37d0:	f94043e0 	ldr	x0, [sp, #128]
    37d4:	94000284 	bl	41e4 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    37d8:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    37dc:	91010021 	add	x1, x1, #0x40
    37e0:	f94043e0 	ldr	x0, [sp, #128]
    37e4:	940002c4 	bl	42f4 <sptPrintElapsedTime>
    37e8:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    37ec:	f94043e0 	ldr	x0, [sp, #128]
    37f0:	940002ea 	bl	4398 <sptFreeTimer>
	sptFreeValueVector(&scratch);
    37f4:	910223e0 	add	x0, sp, #0x88
    37f8:	97fffa74 	bl	21c8 <sptFreeValueVector>
	printf("[Total time]: %lf\n", total_time);
    37fc:	1e604100 	fmov	d0, d8
    3800:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    3804:	9101e000 	add	x0, x0, #0x78
    3808:	97fff6ea 	bl	13b0 <printf@plt>
	printf("\n");
    380c:	52800140 	mov	w0, #0xa                   	// #10
    3810:	97fff6f4 	bl	13e0 <putchar@plt>
	return 0;
    3814:	52800000 	mov	w0, #0x0                   	// #0
    3818:	a94153f3 	ldp	x19, x20, [sp, #16]
    381c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3820:	a94573fb 	ldp	x27, x28, [sp, #80]
    3824:	fd4033e8 	ldr	d8, [sp, #96]
}
    3828:	a9425bf5 	ldp	x21, x22, [sp, #32]
    382c:	a94363f7 	ldp	x23, x24, [sp, #48]
    3830:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    3834:	d65f03c0 	ret
		sptAssert(sptMTTKRP_3D(X, mats, mats_order, mode) == 0);
    3838:	97fffed0 	bl	3378 <sptMTTKRP_3D>
    383c:	34ffff60 	cbz	w0, 3828 <sptMTTKRP+0x260>
    3840:	a90153f3 	stp	x19, x20, [sp, #16]
    3844:	a9046bf9 	stp	x25, x26, [sp, #64]
    3848:	a90573fb 	stp	x27, x28, [sp, #80]
    384c:	fd0033e8 	str	d8, [sp, #96]
    3850:	12800000 	mov	w0, #0xffffffff            	// #-1
    3854:	97fff64b 	bl	1180 <exit@plt>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3858:	b0000004 	adrp	x4, 4000 <sptOmpMTTKRP+0x13c>
    385c:	913fc084 	add	x4, x4, #0xff0
    3860:	528007c3 	mov	w3, #0x3e                  	// #62
    3864:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    3868:	91006042 	add	x2, x2, #0x18
    386c:	52800041 	mov	w1, #0x2                   	// #2
    3870:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    3874:	91010000 	add	x0, x0, #0x40
    3878:	9400033d 	bl	456c <spt_ComplainError>
    387c:	52800040 	mov	w0, #0x2                   	// #2
    3880:	17ffffea 	b	3828 <sptMTTKRP+0x260>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3884:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7b8>
    3888:	91016084 	add	x4, x4, #0x58
    388c:	52800823 	mov	w3, #0x41                  	// #65
    3890:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    3894:	91006042 	add	x2, x2, #0x18
    3898:	52800041 	mov	w1, #0x2                   	// #2
    389c:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    38a0:	91010000 	add	x0, x0, #0x40
    38a4:	94000332 	bl	456c <spt_ComplainError>
    38a8:	52800040 	mov	w0, #0x2                   	// #2
    38ac:	17ffffdf 	b	3828 <sptMTTKRP+0x260>

00000000000038b0 <sptOmpMTTKRP._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    38b0:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    38b4:	910003fd 	mov	x29, sp
    38b8:	a9025bf5 	stp	x21, x22, [sp, #32]
    38bc:	f9400c15 	ldr	x21, [x0, #24]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    38c0:	b4000e95 	cbz	x21, 3a90 <sptOmpMTTKRP._omp_fn.0+0x1e0>
    38c4:	a90153f3 	stp	x19, x20, [sp, #16]
    38c8:	aa0003f3 	mov	x19, x0
    38cc:	97fff679 	bl	12b0 <omp_get_num_threads@plt>
    38d0:	93407c14 	sxtw	x20, w0
    38d4:	97fff643 	bl	11e0 <omp_get_thread_num@plt>
    38d8:	93407c00 	sxtw	x0, w0
    38dc:	9ad40aa1 	udiv	x1, x21, x20
    38e0:	9b14d422 	msub	x2, x1, x20, x21
    38e4:	eb02001f 	cmp	x0, x2
    38e8:	54001103 	b.cc	3b08 <sptOmpMTTKRP._omp_fn.0+0x258>  // b.lo, b.ul, b.last
    38ec:	9b000820 	madd	x0, x1, x0, x2
    38f0:	f9003fe0 	str	x0, [sp, #120]
    38f4:	8b000021 	add	x1, x1, x0
    38f8:	f90047e1 	str	x1, [sp, #136]
    38fc:	eb01001f 	cmp	x0, x1
    3900:	540010a2 	b.cs	3b14 <sptOmpMTTKRP._omp_fn.0+0x264>  // b.hs, b.nlast
    3904:	a90363f7 	stp	x23, x24, [sp, #48]
    3908:	a9046bf9 	stp	x25, x26, [sp, #64]
    390c:	a90573fb 	stp	x27, x28, [sp, #80]
    3910:	6d0627e8 	stp	d8, d9, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    3914:	f9401a60 	ldr	x0, [x19, #48]
    3918:	f9004be0 	str	x0, [sp, #144]
    391c:	f9401660 	ldr	x0, [x19, #40]
    3920:	f9004fe0 	str	x0, [sp, #152]
    3924:	b9404276 	ldr	w22, [x19, #64]
    3928:	b9403e7c 	ldr	w28, [x19, #60]
    392c:	f9401260 	ldr	x0, [x19, #32]
    3930:	f90053e0 	str	x0, [sp, #160]
    3934:	b9403a7b 	ldr	w27, [x19, #56]
    3938:	f9400a7a 	ldr	x26, [x19, #16]
    393c:	f9400679 	ldr	x25, [x19, #8]
    3940:	f9400278 	ldr	x24, [x19]
    3944:	510006c0 	sub	w0, w22, #0x1
    3948:	91000400 	add	x0, x0, #0x1
    394c:	d37ef400 	lsl	x0, x0, #2
    3950:	f9003be0 	str	x0, [sp, #112]
		sptValueVector scratch;  // Temporary array
		sptNewValueVector(&scratch, R, R);
    3954:	2a1603e0 	mov	w0, w22
    3958:	f90057e0 	str	x0, [sp, #168]
    395c:	9102e3e0 	add	x0, sp, #0xb8
    3960:	f90043e0 	str	x0, [sp, #128]
		sptConstantValueVector(&scratch, 0);
    3964:	0f000409 	movi	v9.2s, #0x0
    3968:	14000021 	b	39ec <sptOmpMTTKRP._omp_fn.0+0x13c>
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
			}
		}

		sptIndex const mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    396c:	f9404fe0 	ldr	x0, [sp, #152]
    3970:	f9403fe1 	ldr	x1, [sp, #120]
    3974:	b8617817 	ldr	w23, [x0, x1, lsl #2]
    3978:	1b177f97 	mul	w23, w28, w23
		for(sptIndex r=0; r<R; ++r) {
    397c:	f9404be0 	ldr	x0, [sp, #144]
    3980:	8b170817 	add	x23, x0, x23, lsl #2
    3984:	d2800013 	mov	x19, #0x0                   	// #0
    3988:	34000236 	cbz	w22, 39cc <sptOmpMTTKRP._omp_fn.0+0x11c>
#pragma omp atomic update
			mvals_row[r] += scratch.data[r];
    398c:	f94067e0 	ldr	x0, [sp, #200]
    3990:	bc736808 	ldr	s8, [x0, x19]
    3994:	8b1302f5 	add	x21, x23, x19
    3998:	b94002b4 	ldr	w20, [x21]
    399c:	1e270280 	fmov	s0, w20
    39a0:	1e202900 	fadd	s0, s8, s0
    39a4:	aa1503e2 	mov	x2, x21
    39a8:	1e260001 	fmov	w1, s0
    39ac:	2a1403e0 	mov	w0, w20
    39b0:	94000370 	bl	4770 <__aarch64_cas4_acq_rel>
    39b4:	6b00029f 	cmp	w20, w0
    39b8:	54000721 	b.ne	3a9c <sptOmpMTTKRP._omp_fn.0+0x1ec>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    39bc:	91001273 	add	x19, x19, #0x4
    39c0:	f9403be0 	ldr	x0, [sp, #112]
    39c4:	eb00027f 	cmp	x19, x0
    39c8:	54fffe21 	b.ne	398c <sptOmpMTTKRP._omp_fn.0+0xdc>  // b.any
		}

		sptFreeValueVector(&scratch);
    39cc:	f94043e0 	ldr	x0, [sp, #128]
    39d0:	97fff9fe 	bl	21c8 <sptFreeValueVector>
    39d4:	f9403fe0 	ldr	x0, [sp, #120]
    39d8:	91000400 	add	x0, x0, #0x1
    39dc:	f9003fe0 	str	x0, [sp, #120]
    39e0:	f94047e1 	ldr	x1, [sp, #136]
    39e4:	eb00003f 	cmp	x1, x0
    39e8:	540004a9 	b.ls	3a7c <sptOmpMTTKRP._omp_fn.0+0x1cc>  // b.plast
		sptNewValueVector(&scratch, R, R);
    39ec:	f94057e1 	ldr	x1, [sp, #168]
    39f0:	aa0103e2 	mov	x2, x1
    39f4:	f94043f3 	ldr	x19, [sp, #128]
    39f8:	aa1303e0 	mov	x0, x19
    39fc:	97fff952 	bl	1f44 <sptNewValueVector>
		sptConstantValueVector(&scratch, 0);
    3a00:	1e204120 	fmov	s0, s9
    3a04:	aa1303e0 	mov	x0, x19
    3a08:	97fff97c 	bl	1ff8 <sptConstantValueVector>
		sptMatrix * times_mat = mats[times_mat_index];
    3a0c:	b9400740 	ldr	w0, [x26, #4]
    3a10:	f8607b23 	ldr	x3, [x25, x0, lsl #3]
		sptIndex tmp_i = times_inds[x];
    3a14:	f9403fe2 	ldr	x2, [sp, #120]
    3a18:	d37ef447 	lsl	x7, x2, #2
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3a1c:	f9401301 	ldr	x1, [x24, #32]
    3a20:	8b000400 	add	x0, x0, x0, lsl #1
    3a24:	8b000c20 	add	x0, x1, x0, lsl #3
		sptIndex tmp_i = times_inds[x];
    3a28:	f9400800 	ldr	x0, [x0, #16]
    3a2c:	b8627800 	ldr	w0, [x0, x2, lsl #2]
		sptValue const entry = vals[x];
    3a30:	f94053e1 	ldr	x1, [sp, #160]
    3a34:	bc627821 	ldr	s1, [x1, x2, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3a38:	340001b6 	cbz	w22, 3a6c <sptOmpMTTKRP._omp_fn.0+0x1bc>
			scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
    3a3c:	1b007f80 	mul	w0, w28, w0
    3a40:	0b0002c4 	add	w4, w22, w0
    3a44:	d2800001 	mov	x1, #0x0                   	// #0
    3a48:	f9400862 	ldr	x2, [x3, #16]
    3a4c:	bc605840 	ldr	s0, [x2, w0, uxtw #2]
    3a50:	1e210800 	fmul	s0, s0, s1
    3a54:	f94067e2 	ldr	x2, [sp, #200]
    3a58:	bc216840 	str	s0, [x2, x1]
    3a5c:	11000400 	add	w0, w0, #0x1
    3a60:	91001021 	add	x1, x1, #0x4
    3a64:	6b04001f 	cmp	w0, w4
    3a68:	54ffff01 	b.ne	3a48 <sptOmpMTTKRP._omp_fn.0+0x198>  // b.any
		for(sptIndex i=2; i<nmodes; ++i) {
    3a6c:	d2800046 	mov	x6, #0x2                   	// #2
    3a70:	71000b7f 	cmp	w27, #0x2
    3a74:	540001e8 	b.hi	3ab0 <sptOmpMTTKRP._omp_fn.0+0x200>  // b.pmore
    3a78:	17ffffbd 	b	396c <sptOmpMTTKRP._omp_fn.0+0xbc>
    3a7c:	a94153f3 	ldp	x19, x20, [sp, #16]
    3a80:	a94363f7 	ldp	x23, x24, [sp, #48]
    3a84:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3a88:	a94573fb 	ldp	x27, x28, [sp, #80]
    3a8c:	6d4627e8 	ldp	d8, d9, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    3a90:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3a94:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    3a98:	d65f03c0 	ret
			mvals_row[r] += scratch.data[r];
    3a9c:	2a0003f4 	mov	w20, w0
    3aa0:	17ffffbf 	b	399c <sptOmpMTTKRP._omp_fn.0+0xec>
		for(sptIndex i=2; i<nmodes; ++i) {
    3aa4:	910004c6 	add	x6, x6, #0x1
    3aa8:	6b06037f 	cmp	w27, w6
    3aac:	54fff609 	b.ls	396c <sptOmpMTTKRP._omp_fn.0+0xbc>  // b.plast
			times_mat = mats[times_mat_index];
    3ab0:	b8667b40 	ldr	w0, [x26, x6, lsl #2]
    3ab4:	f8607b24 	ldr	x4, [x25, x0, lsl #3]
			times_inds = X->inds[times_mat_index].data;
    3ab8:	f9401301 	ldr	x1, [x24, #32]
    3abc:	8b000400 	add	x0, x0, x0, lsl #1
    3ac0:	8b000c20 	add	x0, x1, x0, lsl #3
			tmp_i = times_inds[x];
    3ac4:	f9400800 	ldr	x0, [x0, #16]
    3ac8:	b8676800 	ldr	w0, [x0, x7]
			for(sptIndex r=0; r<R; ++r) {
    3acc:	34fffed6 	cbz	w22, 3aa4 <sptOmpMTTKRP._omp_fn.0+0x1f4>
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3ad0:	1b007f80 	mul	w0, w28, w0
    3ad4:	0b0002c5 	add	w5, w22, w0
    3ad8:	d2800001 	mov	x1, #0x0                   	// #0
    3adc:	f94067e2 	ldr	x2, [sp, #200]
    3ae0:	f9400883 	ldr	x3, [x4, #16]
    3ae4:	bc616840 	ldr	s0, [x2, x1]
    3ae8:	bc605861 	ldr	s1, [x3, w0, uxtw #2]
    3aec:	1e210800 	fmul	s0, s0, s1
    3af0:	bc216840 	str	s0, [x2, x1]
    3af4:	91001021 	add	x1, x1, #0x4
    3af8:	11000400 	add	w0, w0, #0x1
    3afc:	6b05001f 	cmp	w0, w5
    3b00:	54fffee1 	b.ne	3adc <sptOmpMTTKRP._omp_fn.0+0x22c>  // b.any
    3b04:	17ffffe8 	b	3aa4 <sptOmpMTTKRP._omp_fn.0+0x1f4>
    3b08:	91000421 	add	x1, x1, #0x1
#pragma omp parallel for schedule(static) num_threads(tk)
    3b0c:	d2800002 	mov	x2, #0x0                   	// #0
    3b10:	17ffff77 	b	38ec <sptOmpMTTKRP._omp_fn.0+0x3c>
    3b14:	a94153f3 	ldp	x19, x20, [sp, #16]
    3b18:	17ffffde 	b	3a90 <sptOmpMTTKRP._omp_fn.0+0x1e0>

0000000000003b1c <sptOmpMTTKRP_3D._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    3b1c:	a9b57bfd 	stp	x29, x30, [sp, #-176]!
    3b20:	910003fd 	mov	x29, sp
    3b24:	a9025bf5 	stp	x21, x22, [sp, #32]
    3b28:	f9400015 	ldr	x21, [x0]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3b2c:	b4000ad5 	cbz	x21, 3c84 <sptOmpMTTKRP_3D._omp_fn.0+0x168>
    3b30:	a90153f3 	stp	x19, x20, [sp, #16]
    3b34:	a9046bf9 	stp	x25, x26, [sp, #64]
    3b38:	aa0003f3 	mov	x19, x0
    3b3c:	97fff5dd 	bl	12b0 <omp_get_num_threads@plt>
    3b40:	93407c14 	sxtw	x20, w0
    3b44:	97fff5a7 	bl	11e0 <omp_get_thread_num@plt>
    3b48:	93407c19 	sxtw	x25, w0
    3b4c:	9ad40aa0 	udiv	x0, x21, x20
    3b50:	9b14d401 	msub	x1, x0, x20, x21
    3b54:	eb01033f 	cmp	x25, x1
    3b58:	54000a03 	b.cc	3c98 <sptOmpMTTKRP_3D._omp_fn.0+0x17c>  // b.lo, b.ul, b.last
    3b5c:	9b190419 	madd	x25, x0, x25, x1
    3b60:	8b190000 	add	x0, x0, x25
    3b64:	f90043e0 	str	x0, [sp, #128]
    3b68:	eb00033f 	cmp	x25, x0
    3b6c:	540009c2 	b.cs	3ca4 <sptOmpMTTKRP_3D._omp_fn.0+0x188>  // b.hs, b.nlast
    3b70:	a90363f7 	stp	x23, x24, [sp, #48]
    3b74:	a90573fb 	stp	x27, x28, [sp, #80]
    3b78:	6d0627e8 	stp	d8, d9, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    3b7c:	f9401e60 	ldr	x0, [x19, #56]
    3b80:	f90047e0 	str	x0, [sp, #136]
    3b84:	f9401a7b 	ldr	x27, [x19, #48]
    3b88:	f9401660 	ldr	x0, [x19, #40]
    3b8c:	f9004be0 	str	x0, [sp, #144]
    3b90:	f940127a 	ldr	x26, [x19, #32]
    3b94:	f9400e60 	ldr	x0, [x19, #24]
    3b98:	f90057e0 	str	x0, [sp, #168]
    3b9c:	f9400a60 	ldr	x0, [x19, #16]
    3ba0:	f9004fe0 	str	x0, [sp, #152]
    3ba4:	b9404660 	ldr	w0, [x19, #68]
    3ba8:	b9007fe0 	str	w0, [sp, #124]
    3bac:	b940427c 	ldr	w28, [x19, #64]
    3bb0:	f9400660 	ldr	x0, [x19, #8]
    3bb4:	f90053e0 	str	x0, [sp, #160]
    3bb8:	14000005 	b	3bcc <sptOmpMTTKRP_3D._omp_fn.0+0xb0>
    3bbc:	91000739 	add	x25, x25, #0x1
    3bc0:	f94043e0 	ldr	x0, [sp, #128]
    3bc4:	eb19001f 	cmp	x0, x25
    3bc8:	54000540 	b.eq	3c70 <sptOmpMTTKRP_3D._omp_fn.0+0x154>  // b.none
		sptIndex mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    3bcc:	f9404fe0 	ldr	x0, [sp, #152]
    3bd0:	b8797814 	ldr	w20, [x0, x25, lsl #2]
    3bd4:	1b147f94 	mul	w20, w28, w20
		sptIndex tmp_i_1 = times_inds_1[x];
    3bd8:	f9404be0 	ldr	x0, [sp, #144]
    3bdc:	b8797817 	ldr	w23, [x0, x25, lsl #2]
		sptIndex tmp_i_2 = times_inds_2[x];
    3be0:	f94047e0 	ldr	x0, [sp, #136]
    3be4:	b8797800 	ldr	w0, [x0, x25, lsl #2]
		sptValue entry = vals[x];
    3be8:	f94053e1 	ldr	x1, [sp, #160]
    3bec:	bc797829 	ldr	s9, [x1, x25, lsl #2]

		for(sptIndex r=0; r<R; ++r) {
    3bf0:	b9407fe1 	ldr	w1, [sp, #124]
    3bf4:	34fffe41 	cbz	w1, 3bbc <sptOmpMTTKRP_3D._omp_fn.0+0xa0>
#pragma omp atomic update
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3bf8:	1b007f80 	mul	w0, w28, w0
    3bfc:	1b177f97 	mul	w23, w28, w23
    3c00:	f94057e1 	ldr	x1, [sp, #168]
    3c04:	8b140834 	add	x20, x1, x20, lsl #2
    3c08:	b9407fe1 	ldr	w1, [sp, #124]
    3c0c:	0b000038 	add	w24, w1, w0
    3c10:	2a0003f3 	mov	w19, w0
    3c14:	4b0002f7 	sub	w23, w23, w0
    3c18:	0b1302e0 	add	w0, w23, w19
    3c1c:	f9400b41 	ldr	x1, [x26, #16]
    3c20:	bc607828 	ldr	s8, [x1, x0, lsl #2]
    3c24:	1e280928 	fmul	s8, s9, s8
    3c28:	f9400b60 	ldr	x0, [x27, #16]
    3c2c:	bc735800 	ldr	s0, [x0, w19, uxtw #2]
    3c30:	1e200908 	fmul	s8, s8, s0
    3c34:	aa1403f6 	mov	x22, x20
    3c38:	b9400295 	ldr	w21, [x20]
    3c3c:	1e2702a0 	fmov	s0, w21
    3c40:	1e202900 	fadd	s0, s8, s0
    3c44:	aa1603e2 	mov	x2, x22
    3c48:	1e260001 	fmov	w1, s0
    3c4c:	2a1503e0 	mov	w0, w21
    3c50:	940002c8 	bl	4770 <__aarch64_cas4_acq_rel>
    3c54:	6b0002bf 	cmp	w21, w0
    3c58:	540001c1 	b.ne	3c90 <sptOmpMTTKRP_3D._omp_fn.0+0x174>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    3c5c:	11000673 	add	w19, w19, #0x1
    3c60:	91001294 	add	x20, x20, #0x4
    3c64:	6b18027f 	cmp	w19, w24
    3c68:	54fffd81 	b.ne	3c18 <sptOmpMTTKRP_3D._omp_fn.0+0xfc>  // b.any
    3c6c:	17ffffd4 	b	3bbc <sptOmpMTTKRP_3D._omp_fn.0+0xa0>
    3c70:	a94153f3 	ldp	x19, x20, [sp, #16]
    3c74:	a94363f7 	ldp	x23, x24, [sp, #48]
    3c78:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3c7c:	a94573fb 	ldp	x27, x28, [sp, #80]
    3c80:	6d4627e8 	ldp	d8, d9, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    3c84:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3c88:	a8cb7bfd 	ldp	x29, x30, [sp], #176
    3c8c:	d65f03c0 	ret
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3c90:	2a0003f5 	mov	w21, w0
    3c94:	17ffffea 	b	3c3c <sptOmpMTTKRP_3D._omp_fn.0+0x120>
    3c98:	91000400 	add	x0, x0, #0x1
#pragma omp parallel for schedule(static) num_threads(tk)
    3c9c:	d2800001 	mov	x1, #0x0                   	// #0
    3ca0:	17ffffaf 	b	3b5c <sptOmpMTTKRP_3D._omp_fn.0+0x40>
    3ca4:	a94153f3 	ldp	x19, x20, [sp, #16]
    3ca8:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3cac:	17fffff6 	b	3c84 <sptOmpMTTKRP_3D._omp_fn.0+0x168>

0000000000003cb0 <sptOmpMTTKRP_3D>:
{
    3cb0:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    3cb4:	910003fd 	mov	x29, sp
    3cb8:	a90153f3 	stp	x19, x20, [sp, #16]
    3cbc:	aa0003f4 	mov	x20, x0
    3cc0:	b9007fe4 	str	w4, [sp, #124]
	sptIndex const * const ndims = X->ndims;
    3cc4:	f9400805 	ldr	x5, [x0, #16]
	sptAssert(nmodes ==3);
    3cc8:	b9400000 	ldr	w0, [x0]
    3ccc:	71000c1f 	cmp	w0, #0x3
    3cd0:	54000c01 	b.ne	3e50 <sptOmpMTTKRP_3D+0x1a0>  // b.any
    3cd4:	a9025bf5 	stp	x21, x22, [sp, #32]
    3cd8:	aa0103f3 	mov	x19, x1
    3cdc:	aa0203f5 	mov	x21, x2
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3ce0:	f9400c26 	ldr	x6, [x1, #24]
    3ce4:	b94004c4 	ldr	w4, [x6, #4]
    3ce8:	d2800000 	mov	x0, #0x0                   	// #0
    3cec:	f8607a61 	ldr	x1, [x19, x0, lsl #3]
    3cf0:	b9400422 	ldr	w2, [x1, #4]
    3cf4:	6b04005f 	cmp	w2, w4
    3cf8:	54000ba1 	b.ne	3e6c <sptOmpMTTKRP_3D+0x1bc>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    3cfc:	b9400022 	ldr	w2, [x1]
    3d00:	b86078a1 	ldr	w1, [x5, x0, lsl #2]
    3d04:	6b01005f 	cmp	w2, w1
    3d08:	54000c81 	b.ne	3e98 <sptOmpMTTKRP_3D+0x1e8>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    3d0c:	91000400 	add	x0, x0, #0x1
    3d10:	f1000c1f 	cmp	x0, #0x3
    3d14:	54fffec1 	b.ne	3cec <sptOmpMTTKRP_3D+0x3c>  // b.any
    3d18:	a90363f7 	stp	x23, x24, [sp, #48]
    3d1c:	a9046bf9 	stp	x25, x26, [sp, #64]
    3d20:	a90573fb 	stp	x27, x28, [sp, #80]
    3d24:	fd0033e8 	str	d8, [sp, #96]
	sptNnzIndex const nnz = X->nnz;
    3d28:	f9400e98 	ldr	x24, [x20, #24]
	sptValue const * const restrict vals = X->values.data;
    3d2c:	f9401e99 	ldr	x25, [x20, #56]
	sptIndex const stride = mats[0]->stride;
    3d30:	f9400260 	ldr	x0, [x19]
    3d34:	b9400c16 	ldr	w22, [x0, #12]
	sptIndex const tmpI = mats[mode]->nrows;
    3d38:	2a0303e3 	mov	w3, w3
    3d3c:	f8637a60 	ldr	x0, [x19, x3, lsl #3]
	sptIndex const R = mats[mode]->ncols;
    3d40:	b940041a 	ldr	w26, [x0, #4]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d44:	f9401281 	ldr	x1, [x20, #32]
    3d48:	8b030463 	add	x3, x3, x3, lsl #1
    3d4c:	8b030c23 	add	x3, x1, x3, lsl #3
    3d50:	f940087b 	ldr	x27, [x3, #16]
	sptValue * const restrict mvals = mats[nmodes]->values;
    3d54:	f94008d7 	ldr	x23, [x6, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d58:	b9400002 	ldr	w2, [x0]
    3d5c:	1b027ec2 	mul	w2, w22, w2
    3d60:	d37ef442 	lsl	x2, x2, #2
    3d64:	52800001 	mov	w1, #0x0                   	// #0
    3d68:	aa1703e0 	mov	x0, x23
    3d6c:	97fff541 	bl	1270 <memset@plt>
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    3d70:	b94006a1 	ldr	w1, [x21, #4]
    3d74:	f8617a7c 	ldr	x28, [x19, x1, lsl #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    3d78:	f9401282 	ldr	x2, [x20, #32]
    3d7c:	8b010421 	add	x1, x1, x1, lsl #1
    3d80:	8b010c41 	add	x1, x2, x1, lsl #3
    3d84:	f9400834 	ldr	x20, [x1, #16]
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    3d88:	b9400aa0 	ldr	w0, [x21, #8]
    3d8c:	f8607a73 	ldr	x19, [x19, x0, lsl #3]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    3d90:	8b000400 	add	x0, x0, x0, lsl #1
    3d94:	8b000c40 	add	x0, x2, x0, lsl #3
    3d98:	f9400815 	ldr	x21, [x0, #16]
	sptNewTimer(&timer, 0);
    3d9c:	52800001 	mov	w1, #0x0                   	// #0
    3da0:	910323e0 	add	x0, sp, #0xc8
    3da4:	940000c6 	bl	40bc <sptNewTimer>
	sptStartTimer(timer);
    3da8:	f94067e0 	ldr	x0, [sp, #200]
    3dac:	940000de 	bl	4124 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    3db0:	f9005ff5 	str	x21, [sp, #184]
    3db4:	f9005bf3 	str	x19, [sp, #176]
    3db8:	f90057f4 	str	x20, [sp, #168]
    3dbc:	f90053fc 	str	x28, [sp, #160]
    3dc0:	f9004ff7 	str	x23, [sp, #152]
    3dc4:	f9004bfb 	str	x27, [sp, #144]
    3dc8:	b900c7fa 	str	w26, [sp, #196]
    3dcc:	b900c3f6 	str	w22, [sp, #192]
    3dd0:	f90047f9 	str	x25, [sp, #136]
    3dd4:	f90043f8 	str	x24, [sp, #128]
    3dd8:	52800003 	mov	w3, #0x0                   	// #0
    3ddc:	b9407fe2 	ldr	w2, [sp, #124]
    3de0:	910203e1 	add	x1, sp, #0x80
    3de4:	90000000 	adrp	x0, 3000 <sptRandomizeMatrix+0x4>
    3de8:	912c7000 	add	x0, x0, #0xb1c
    3dec:	97fff579 	bl	13d0 <GOMP_parallel@plt>
		}
	}
	sptStopTimer(timer);
    3df0:	f94067e0 	ldr	x0, [sp, #200]
    3df4:	940000fc 	bl	41e4 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    3df8:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    3dfc:	9102e021 	add	x1, x1, #0xb8
    3e00:	f94067e0 	ldr	x0, [sp, #200]
    3e04:	9400013c 	bl	42f4 <sptPrintElapsedTime>
    3e08:	1e604008 	fmov	d8, d0

	sptFreeTimer(timer);
    3e0c:	f94067e0 	ldr	x0, [sp, #200]
    3e10:	94000162 	bl	4398 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    3e14:	1e604100 	fmov	d0, d8
    3e18:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    3e1c:	9101e000 	add	x0, x0, #0x78
    3e20:	97fff564 	bl	13b0 <printf@plt>
	printf("\n");
    3e24:	52800140 	mov	w0, #0xa                   	// #10
    3e28:	97fff56e 	bl	13e0 <putchar@plt>

	return 0;
    3e2c:	52800000 	mov	w0, #0x0                   	// #0
    3e30:	a94363f7 	ldp	x23, x24, [sp, #48]
    3e34:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3e38:	a94573fb 	ldp	x27, x28, [sp, #80]
    3e3c:	fd4033e8 	ldr	d8, [sp, #96]
}
    3e40:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3e44:	a94153f3 	ldp	x19, x20, [sp, #16]
    3e48:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    3e4c:	d65f03c0 	ret
    3e50:	a9025bf5 	stp	x21, x22, [sp, #32]
    3e54:	a90363f7 	stp	x23, x24, [sp, #48]
    3e58:	a9046bf9 	stp	x25, x26, [sp, #64]
    3e5c:	a90573fb 	stp	x27, x28, [sp, #80]
    3e60:	fd0033e8 	str	d8, [sp, #96]
	sptAssert(nmodes ==3);
    3e64:	12800000 	mov	w0, #0xffffffff            	// #-1
    3e68:	97fff4c6 	bl	1180 <exit@plt>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3e6c:	b0000004 	adrp	x4, 4000 <sptOmpMTTKRP+0x13c>
    3e70:	913fc084 	add	x4, x4, #0xff0
    3e74:	528012c3 	mov	w3, #0x96                  	// #150
    3e78:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    3e7c:	91024042 	add	x2, x2, #0x90
    3e80:	52800041 	mov	w1, #0x2                   	// #2
    3e84:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    3e88:	9102e000 	add	x0, x0, #0xb8
    3e8c:	940001b8 	bl	456c <spt_ComplainError>
    3e90:	52800040 	mov	w0, #0x2                   	// #2
    3e94:	17ffffeb 	b	3e40 <sptOmpMTTKRP_3D+0x190>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3e98:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7b8>
    3e9c:	91016084 	add	x4, x4, #0x58
    3ea0:	52801323 	mov	w3, #0x99                  	// #153
    3ea4:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    3ea8:	91024042 	add	x2, x2, #0x90
    3eac:	52800041 	mov	w1, #0x2                   	// #2
    3eb0:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    3eb4:	9102e000 	add	x0, x0, #0xb8
    3eb8:	940001ad 	bl	456c <spt_ComplainError>
    3ebc:	52800040 	mov	w0, #0x2                   	// #2
    3ec0:	17ffffe0 	b	3e40 <sptOmpMTTKRP_3D+0x190>

0000000000003ec4 <sptOmpMTTKRP>:
{
    3ec4:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    3ec8:	910003fd 	mov	x29, sp
    3ecc:	a90153f3 	stp	x19, x20, [sp, #16]
    3ed0:	a9025bf5 	stp	x21, x22, [sp, #32]
    3ed4:	a90363f7 	stp	x23, x24, [sp, #48]
    3ed8:	aa0003f5 	mov	x21, x0
    3edc:	aa0103f3 	mov	x19, x1
    3ee0:	aa0203f7 	mov	x23, x2
    3ee4:	2a0403f6 	mov	w22, w4
	sptIndex const nmodes = X->nmodes;
    3ee8:	b9400014 	ldr	w20, [x0]
	if(nmodes == 3) {
    3eec:	71000e9f 	cmp	w20, #0x3
    3ef0:	54000ac0 	b.eq	4048 <sptOmpMTTKRP+0x184>  // b.none
	sptIndex const * const ndims = X->ndims;
    3ef4:	f9400805 	ldr	x5, [x0, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    3ef8:	340001f4 	cbz	w20, 3f34 <sptOmpMTTKRP+0x70>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3efc:	f8745820 	ldr	x0, [x1, w20, uxtw #3]
    3f00:	b9400404 	ldr	w4, [x0, #4]
    3f04:	d2800000 	mov	x0, #0x0                   	// #0
    3f08:	f8607a61 	ldr	x1, [x19, x0, lsl #3]
    3f0c:	b9400422 	ldr	w2, [x1, #4]
    3f10:	6b04005f 	cmp	w2, w4
    3f14:	54000a81 	b.ne	4064 <sptOmpMTTKRP+0x1a0>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    3f18:	b9400022 	ldr	w2, [x1]
    3f1c:	b86078a1 	ldr	w1, [x5, x0, lsl #2]
    3f20:	6b01005f 	cmp	w2, w1
    3f24:	54000b61 	b.ne	4090 <sptOmpMTTKRP+0x1cc>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    3f28:	91000400 	add	x0, x0, #0x1
    3f2c:	6b00029f 	cmp	w20, w0
    3f30:	54fffec8 	b.hi	3f08 <sptOmpMTTKRP+0x44>  // b.pmore
    3f34:	a9046bf9 	stp	x25, x26, [sp, #64]
    3f38:	a90573fb 	stp	x27, x28, [sp, #80]
    3f3c:	fd0033e8 	str	d8, [sp, #96]
	sptNnzIndex const nnz = X->nnz;
    3f40:	f9400ea0 	ldr	x0, [x21, #24]
    3f44:	f9003fe0 	str	x0, [sp, #120]
	sptValue const * const vals = X->values.data;
    3f48:	f9401eba 	ldr	x26, [x21, #56]
	sptIndex const stride = mats[0]->stride;
    3f4c:	f9400260 	ldr	x0, [x19]
    3f50:	b9400c18 	ldr	w24, [x0, #12]
	sptIndex const tmpI = mats[mode]->nrows;
    3f54:	2a0303e3 	mov	w3, w3
    3f58:	f8637a60 	ldr	x0, [x19, x3, lsl #3]
	sptIndex const R = mats[mode]->ncols;
    3f5c:	b940041b 	ldr	w27, [x0, #4]
	sptIndex const * const mode_ind = X->inds[mode].data;
    3f60:	f94012a1 	ldr	x1, [x21, #32]
    3f64:	8b030463 	add	x3, x3, x3, lsl #1
    3f68:	8b030c23 	add	x3, x1, x3, lsl #3
    3f6c:	f940087c 	ldr	x28, [x3, #16]
	sptValue * const restrict mvals = mats[nmodes]->values;
    3f70:	f8745a61 	ldr	x1, [x19, w20, uxtw #3]
    3f74:	f9400839 	ldr	x25, [x1, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3f78:	b9400002 	ldr	w2, [x0]
    3f7c:	1b027f02 	mul	w2, w24, w2
    3f80:	d37ef442 	lsl	x2, x2, #2
    3f84:	52800001 	mov	w1, #0x0                   	// #0
    3f88:	aa1903e0 	mov	x0, x25
    3f8c:	97fff4b9 	bl	1270 <memset@plt>
	sptNewTimer(&timer, 0);
    3f90:	52800001 	mov	w1, #0x0                   	// #0
    3f94:	910323e0 	add	x0, sp, #0xc8
    3f98:	94000049 	bl	40bc <sptNewTimer>
	sptStartTimer(timer);
    3f9c:	f94067e0 	ldr	x0, [sp, #200]
    3fa0:	94000061 	bl	4124 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    3fa4:	f9005bf9 	str	x25, [sp, #176]
    3fa8:	f90057fc 	str	x28, [sp, #168]
    3fac:	b900c3fb 	str	w27, [sp, #192]
    3fb0:	b900bff8 	str	w24, [sp, #188]
    3fb4:	f90053fa 	str	x26, [sp, #160]
    3fb8:	f9403fe0 	ldr	x0, [sp, #120]
    3fbc:	f9004fe0 	str	x0, [sp, #152]
    3fc0:	b900bbf4 	str	w20, [sp, #184]
    3fc4:	f9004bf7 	str	x23, [sp, #144]
    3fc8:	f90047f3 	str	x19, [sp, #136]
    3fcc:	f90043f5 	str	x21, [sp, #128]
    3fd0:	52800003 	mov	w3, #0x0                   	// #0
    3fd4:	2a1603e2 	mov	w2, w22
    3fd8:	910203e1 	add	x1, sp, #0x80
    3fdc:	90000000 	adrp	x0, 3000 <sptRandomizeMatrix+0x4>
    3fe0:	9122c000 	add	x0, x0, #0x8b0
    3fe4:	97fff4fb 	bl	13d0 <GOMP_parallel@plt>
	sptStopTimer(timer);
    3fe8:	f94067e0 	ldr	x0, [sp, #200]
    3fec:	9400007e 	bl	41e4 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    3ff0:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    3ff4:	9102e021 	add	x1, x1, #0xb8
    3ff8:	f94067e0 	ldr	x0, [sp, #200]
    3ffc:	940000be 	bl	42f4 <sptPrintElapsedTime>
    4000:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    4004:	f94067e0 	ldr	x0, [sp, #200]
    4008:	940000e4 	bl	4398 <sptFreeTimer>
	printf("[Total time]: %lf\n", total_time);
    400c:	1e604100 	fmov	d0, d8
    4010:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    4014:	9101e000 	add	x0, x0, #0x78
    4018:	97fff4e6 	bl	13b0 <printf@plt>
	printf("\n");
    401c:	52800140 	mov	w0, #0xa                   	// #10
    4020:	97fff4f0 	bl	13e0 <putchar@plt>
	return 0;
    4024:	52800000 	mov	w0, #0x0                   	// #0
    4028:	a9446bf9 	ldp	x25, x26, [sp, #64]
    402c:	a94573fb 	ldp	x27, x28, [sp, #80]
    4030:	fd4033e8 	ldr	d8, [sp, #96]
}
    4034:	a94153f3 	ldp	x19, x20, [sp, #16]
    4038:	a9425bf5 	ldp	x21, x22, [sp, #32]
    403c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4040:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    4044:	d65f03c0 	ret
		sptAssert(sptOmpMTTKRP_3D(X, mats, mats_order, mode, tk) == 0);
    4048:	97ffff1a 	bl	3cb0 <sptOmpMTTKRP_3D>
    404c:	34ffff40 	cbz	w0, 4034 <sptOmpMTTKRP+0x170>
    4050:	a9046bf9 	stp	x25, x26, [sp, #64]
    4054:	a90573fb 	stp	x27, x28, [sp, #80]
    4058:	fd0033e8 	str	d8, [sp, #96]
    405c:	12800000 	mov	w0, #0xffffffff            	// #-1
    4060:	97fff448 	bl	1180 <exit@plt>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    4064:	90000004 	adrp	x4, 4000 <sptOmpMTTKRP+0x13c>
    4068:	913fc084 	add	x4, x4, #0xff0
    406c:	52800823 	mov	w3, #0x41                  	// #65
    4070:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    4074:	91024042 	add	x2, x2, #0x90
    4078:	52800041 	mov	w1, #0x2                   	// #2
    407c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    4080:	9102e000 	add	x0, x0, #0xb8
    4084:	9400013a 	bl	456c <spt_ComplainError>
    4088:	52800040 	mov	w0, #0x2                   	// #2
    408c:	17ffffea 	b	4034 <sptOmpMTTKRP+0x170>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4090:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7b8>
    4094:	91016084 	add	x4, x4, #0x58
    4098:	52800883 	mov	w3, #0x44                  	// #68
    409c:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    40a0:	91024042 	add	x2, x2, #0x90
    40a4:	52800041 	mov	w1, #0x2                   	// #2
    40a8:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    40ac:	9102e000 	add	x0, x0, #0xb8
    40b0:	9400012f 	bl	456c <spt_ComplainError>
    40b4:	52800040 	mov	w0, #0x2                   	// #2
    40b8:	17ffffdf 	b	4034 <sptOmpMTTKRP+0x170>

00000000000040bc <sptNewTimer>:
		int use_cuda;
		struct timespec start_timespec;
		struct timespec stop_timespec;
};

int sptNewTimer(sptTimer *timer, int use_cuda) {
    40bc:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    40c0:	910003fd 	mov	x29, sp
    40c4:	a90153f3 	stp	x19, x20, [sp, #16]
    40c8:	aa0003f4 	mov	x20, x0
    40cc:	2a0103f3 	mov	w19, w1
	*timer = (sptTimer) malloc(sizeof **timer);
    40d0:	d2800500 	mov	x0, #0x28                  	// #40
    40d4:	97fff453 	bl	1220 <malloc@plt>
    40d8:	f9000280 	str	x0, [x20]
	(*timer)->use_cuda = use_cuda;
    40dc:	b9000013 	str	w19, [x0]
	if(use_cuda) {
    40e0:	350000b3 	cbnz	w19, 40f4 <sptNewTimer+0x38>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	return 0;
}
    40e4:	2a1303e0 	mov	w0, w19
    40e8:	a94153f3 	ldp	x19, x20, [sp, #16]
    40ec:	a8c27bfd 	ldp	x29, x30, [sp], #32
    40f0:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    40f4:	52800073 	mov	w19, #0x3                   	// #3
    40f8:	72a00053 	movk	w19, #0x2, lsl #16
    40fc:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7b8>
    4100:	91034084 	add	x4, x4, #0xd0
    4104:	528004e3 	mov	w3, #0x27                  	// #39
    4108:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    410c:	9103e042 	add	x2, x2, #0xf8
    4110:	2a1303e1 	mov	w1, w19
    4114:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    4118:	91048000 	add	x0, x0, #0x120
    411c:	94000114 	bl	456c <spt_ComplainError>
    4120:	17fffff1 	b	40e4 <sptNewTimer+0x28>

0000000000004124 <sptStartTimer>:

int sptStartTimer(sptTimer timer) {
    4124:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    4128:	910003fd 	mov	x29, sp
    412c:	a90153f3 	stp	x19, x20, [sp, #16]
	int result;
	if(timer->use_cuda) {
    4130:	b9400013 	ldr	w19, [x0]
    4134:	35000113 	cbnz	w19, 4154 <sptStartTimer+0x30>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
    4138:	91002001 	add	x1, x0, #0x8
    413c:	52800020 	mov	w0, #0x1                   	// #1
    4140:	97fff41c 	bl	11b0 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    4144:	35000220 	cbnz	w0, 4188 <sptStartTimer+0x64>
	}
	return 0;
}
    4148:	a94153f3 	ldp	x19, x20, [sp, #16]
    414c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    4150:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4154:	52800073 	mov	w19, #0x3                   	// #3
    4158:	72a00053 	movk	w19, #0x2, lsl #16
    415c:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7b8>
    4160:	91034084 	add	x4, x4, #0xd0
    4164:	528005e3 	mov	w3, #0x2f                  	// #47
    4168:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    416c:	9103e042 	add	x2, x2, #0xf8
    4170:	2a1303e1 	mov	w1, w19
    4174:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    4178:	91048000 	add	x0, x0, #0x120
    417c:	940000fc 	bl	456c <spt_ComplainError>
    4180:	2a1303e0 	mov	w0, w19
    4184:	17fffff1 	b	4148 <sptStartTimer+0x24>
    4188:	f90013f5 	str	x21, [sp, #32]
		spt_CheckOSError(result, "Timer New");
    418c:	97fff48d 	bl	13c0 <__errno_location@plt>
    4190:	aa0003f4 	mov	x20, x0
    4194:	b9400015 	ldr	w21, [x0]
	return 0;
    4198:	2a1303e0 	mov	w0, w19
		spt_CheckOSError(result, "Timer New");
    419c:	314042bf 	cmn	w21, #0x10, lsl #12
    41a0:	54000061 	b.ne	41ac <sptStartTimer+0x88>  // b.any
    41a4:	f94013f5 	ldr	x21, [sp, #32]
    41a8:	17ffffe8 	b	4148 <sptStartTimer+0x24>
    41ac:	2a1503e0 	mov	w0, w21
    41b0:	97fff444 	bl	12c0 <strerror@plt>
    41b4:	aa0003e4 	mov	x4, x0
    41b8:	52800643 	mov	w3, #0x32                  	// #50
    41bc:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    41c0:	9103e042 	add	x2, x2, #0xf8
    41c4:	114042a1 	add	w1, w21, #0x10, lsl #12
    41c8:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    41cc:	91048000 	add	x0, x0, #0x120
    41d0:	940000e7 	bl	456c <spt_ComplainError>
    41d4:	b9400280 	ldr	w0, [x20]
    41d8:	11404000 	add	w0, w0, #0x10, lsl #12
    41dc:	f94013f5 	ldr	x21, [sp, #32]
    41e0:	17ffffda 	b	4148 <sptStartTimer+0x24>

00000000000041e4 <sptStopTimer>:

int sptStopTimer(sptTimer timer) {
    41e4:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    41e8:	910003fd 	mov	x29, sp
    41ec:	a90153f3 	stp	x19, x20, [sp, #16]
	int result;
	if(timer->use_cuda) {
    41f0:	b9400013 	ldr	w19, [x0]
    41f4:	35000113 	cbnz	w19, 4214 <sptStopTimer+0x30>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->stop_timespec);
    41f8:	91006001 	add	x1, x0, #0x18
    41fc:	52800020 	mov	w0, #0x1                   	// #1
    4200:	97fff3ec 	bl	11b0 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    4204:	35000220 	cbnz	w0, 4248 <sptStopTimer+0x64>
	}
	return 0;
}
    4208:	a94153f3 	ldp	x19, x20, [sp, #16]
    420c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    4210:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4214:	52800073 	mov	w19, #0x3                   	// #3
    4218:	72a00053 	movk	w19, #0x2, lsl #16
    421c:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7b8>
    4220:	91034084 	add	x4, x4, #0xd0
    4224:	52800743 	mov	w3, #0x3a                  	// #58
    4228:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    422c:	9103e042 	add	x2, x2, #0xf8
    4230:	2a1303e1 	mov	w1, w19
    4234:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    4238:	91048000 	add	x0, x0, #0x120
    423c:	940000cc 	bl	456c <spt_ComplainError>
    4240:	2a1303e0 	mov	w0, w19
    4244:	17fffff1 	b	4208 <sptStopTimer+0x24>
    4248:	f90013f5 	str	x21, [sp, #32]
		spt_CheckOSError(result, "Timer New");
    424c:	97fff45d 	bl	13c0 <__errno_location@plt>
    4250:	aa0003f4 	mov	x20, x0
    4254:	b9400015 	ldr	w21, [x0]
	return 0;
    4258:	2a1303e0 	mov	w0, w19
		spt_CheckOSError(result, "Timer New");
    425c:	314042bf 	cmn	w21, #0x10, lsl #12
    4260:	54000061 	b.ne	426c <sptStopTimer+0x88>  // b.any
    4264:	f94013f5 	ldr	x21, [sp, #32]
    4268:	17ffffe8 	b	4208 <sptStopTimer+0x24>
    426c:	2a1503e0 	mov	w0, w21
    4270:	97fff414 	bl	12c0 <strerror@plt>
    4274:	aa0003e4 	mov	x4, x0
    4278:	528007a3 	mov	w3, #0x3d                  	// #61
    427c:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    4280:	9103e042 	add	x2, x2, #0xf8
    4284:	114042a1 	add	w1, w21, #0x10, lsl #12
    4288:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    428c:	91048000 	add	x0, x0, #0x120
    4290:	940000b7 	bl	456c <spt_ComplainError>
    4294:	b9400280 	ldr	w0, [x20]
    4298:	11404000 	add	w0, w0, #0x10, lsl #12
    429c:	f94013f5 	ldr	x21, [sp, #32]
    42a0:	17ffffda 	b	4208 <sptStopTimer+0x24>

00000000000042a4 <sptElapsedTime>:

double sptElapsedTime(const sptTimer timer) {
	if(timer->use_cuda) {
    42a4:	b9400001 	ldr	w1, [x0]
    42a8:	35000201 	cbnz	w1, 42e8 <sptElapsedTime+0x44>
		return NAN;
	} else {
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    42ac:	f9401001 	ldr	x1, [x0, #32]
    42b0:	f9400802 	ldr	x2, [x0, #16]
    42b4:	cb020021 	sub	x1, x1, x2
    42b8:	9e670020 	fmov	d0, x1
    42bc:	5e61d800 	scvtf	d0, d0
    42c0:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    42c4:	fd40a021 	ldr	d1, [x1, #320]
    42c8:	1e610800 	fmul	d0, d0, d1
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    42cc:	f9400c01 	ldr	x1, [x0, #24]
    42d0:	f9400400 	ldr	x0, [x0, #8]
    42d4:	cb000020 	sub	x0, x1, x0
    42d8:	9e670001 	fmov	d1, x0
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    42dc:	5e61d821 	scvtf	d1, d1
    42e0:	1e612800 	fadd	d0, d0, d1
	}
}
    42e4:	d65f03c0 	ret
		return NAN;
    42e8:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    42ec:	fd40a400 	ldr	d0, [x0, #328]
    42f0:	17fffffd 	b	42e4 <sptElapsedTime+0x40>

00000000000042f4 <sptPrintElapsedTime>:

double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    42f4:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    42f8:	910003fd 	mov	x29, sp
    42fc:	f9000bf3 	str	x19, [sp, #16]
    4300:	fd000fe8 	str	d8, [sp, #24]
    4304:	aa0103f3 	mov	x19, x1
	double elapsed_time = sptElapsedTime(timer);
    4308:	97ffffe7 	bl	42a4 <sptElapsedTime>
    430c:	1e604008 	fmov	d8, d0
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4310:	aa1303e2 	mov	x2, x19
    4314:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    4318:	9104c021 	add	x1, x1, #0x130
    431c:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    4320:	f947e000 	ldr	x0, [x0, #4032]
    4324:	f9400000 	ldr	x0, [x0]
    4328:	97fff432 	bl	13f0 <fprintf@plt>
	return elapsed_time;
}
    432c:	1e604100 	fmov	d0, d8
    4330:	fd400fe8 	ldr	d8, [sp, #24]
    4334:	f9400bf3 	ldr	x19, [sp, #16]
    4338:	a8c27bfd 	ldp	x29, x30, [sp], #32
    433c:	d65f03c0 	ret

0000000000004340 <sptPrintAverageElapsedTime>:


double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    4340:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    4344:	910003fd 	mov	x29, sp
    4348:	a90153f3 	stp	x19, x20, [sp, #16]
    434c:	fd0013e8 	str	d8, [sp, #32]
    4350:	2a0103f4 	mov	w20, w1
    4354:	aa0203f3 	mov	x19, x2
	double elapsed_time = sptElapsedTime(timer) / niters;
    4358:	97ffffd3 	bl	42a4 <sptElapsedTime>
    435c:	1e620288 	scvtf	d8, w20
    4360:	1e681808 	fdiv	d8, d0, d8
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4364:	1e604100 	fmov	d0, d8
    4368:	aa1303e2 	mov	x2, x19
    436c:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    4370:	9104c021 	add	x1, x1, #0x130
    4374:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    4378:	f947e000 	ldr	x0, [x0, #4032]
    437c:	f9400000 	ldr	x0, [x0]
    4380:	97fff41c 	bl	13f0 <fprintf@plt>
	return elapsed_time;
}
    4384:	1e604100 	fmov	d0, d8
    4388:	fd4013e8 	ldr	d8, [sp, #32]
    438c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4390:	a8c37bfd 	ldp	x29, x30, [sp], #48
    4394:	d65f03c0 	ret

0000000000004398 <sptFreeTimer>:


int sptFreeTimer(sptTimer timer) {
    4398:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    439c:	910003fd 	mov	x29, sp
    43a0:	f9000bf3 	str	x19, [sp, #16]
	if(timer->use_cuda) {
    43a4:	b9400013 	ldr	w19, [x0]
    43a8:	350000d3 	cbnz	w19, 43c0 <sptFreeTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	free(timer);
    43ac:	97fff3e9 	bl	1350 <free@plt>
	return 0;
}
    43b0:	2a1303e0 	mov	w0, w19
    43b4:	f9400bf3 	ldr	x19, [sp, #16]
    43b8:	a8c27bfd 	ldp	x29, x30, [sp], #32
    43bc:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    43c0:	52800073 	mov	w19, #0x3                   	// #3
    43c4:	72a00053 	movk	w19, #0x2, lsl #16
    43c8:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7b8>
    43cc:	91034084 	add	x4, x4, #0xd0
    43d0:	52800b63 	mov	w3, #0x5b                  	// #91
    43d4:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    43d8:	9103e042 	add	x2, x2, #0xf8
    43dc:	2a1303e1 	mov	w1, w19
    43e0:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    43e4:	91048000 	add	x0, x0, #0x120
    43e8:	94000061 	bl	456c <spt_ComplainError>
    43ec:	17fffff1 	b	43b0 <sptFreeTimer+0x18>

00000000000043f0 <sptDumpMatrix>:
 *
 * @param mtx   a valid pointer to a sptMatrix variable
 * @param fp a file pointer
 *
 */
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    43f0:	a9b97bfd 	stp	x29, x30, [sp, #-112]!
    43f4:	910003fd 	mov	x29, sp
    43f8:	a90153f3 	stp	x19, x20, [sp, #16]
    43fc:	a9025bf5 	stp	x21, x22, [sp, #32]
    4400:	a90573fb 	stp	x27, x28, [sp, #80]
    4404:	aa0003f6 	mov	x22, x0
    4408:	aa0103f5 	mov	x21, x1
	int iores;
	sptIndex nrows = mtx->nrows;
    440c:	b940001c 	ldr	w28, [x0]
	sptIndex ncols = mtx->ncols;
    4410:	b940041b 	ldr	w27, [x0, #4]
	sptIndex stride = mtx->stride;
    4414:	b9400c00 	ldr	w0, [x0, #12]
    4418:	b9006fe0 	str	w0, [sp, #108]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    441c:	2a1b03e3 	mov	w3, w27
    4420:	2a1c03e2 	mov	w2, w28
    4424:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    4428:	91054021 	add	x1, x1, #0x150
    442c:	aa1503e0 	mov	x0, x21
    4430:	97fff3f0 	bl	13f0 <fprintf@plt>
	spt_CheckOSError(iores < 0, "Mtx Dump");
    4434:	37f801e0 	tbnz	w0, #31, 4470 <sptDumpMatrix+0x80>
    4438:	a90363f7 	stp	x23, x24, [sp, #48]
    443c:	a9046bf9 	stp	x25, x26, [sp, #64]
	for(sptIndex i=0; i < nrows; ++i) {
    4440:	2a1b03f4 	mov	w20, w27
    4444:	5280001a 	mov	w26, #0x0                   	// #0
		for(sptIndex j=0; j < ncols; ++j) {
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4448:	b0000017 	adrp	x23, 5000 <_IO_stdin_used+0x7b8>
    444c:	910662f7 	add	x23, x23, #0x198
	for(sptIndex i=0; i < nrows; ++i) {
    4450:	3500089c 	cbnz	w28, 4560 <sptDumpMatrix+0x170>
			spt_CheckOSError(iores < 0, "Mtx Dump");
		}
		iores = fprintf(fp, "\n");
	}
	iores = fprintf(fp, "\n");
    4454:	aa1503e1 	mov	x1, x21
    4458:	52800140 	mov	w0, #0xa                   	// #10
    445c:	97fff351 	bl	11a0 <fputc@plt>
	return 0;
    4460:	52800000 	mov	w0, #0x0                   	// #0
    4464:	a94363f7 	ldp	x23, x24, [sp, #48]
    4468:	a9446bf9 	ldp	x25, x26, [sp, #64]
    446c:	14000030 	b	452c <sptDumpMatrix+0x13c>
	spt_CheckOSError(iores < 0, "Mtx Dump");
    4470:	97fff3d4 	bl	13c0 <__errno_location@plt>
    4474:	aa0003f3 	mov	x19, x0
    4478:	b9400014 	ldr	w20, [x0]
    447c:	3140429f 	cmn	w20, #0x10, lsl #12
    4480:	54fffdc0 	b.eq	4438 <sptDumpMatrix+0x48>  // b.none
    4484:	2a1403e0 	mov	w0, w20
    4488:	97fff38e 	bl	12c0 <strerror@plt>
    448c:	aa0003e4 	mov	x4, x0
    4490:	52800543 	mov	w3, #0x2a                  	// #42
    4494:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    4498:	91058042 	add	x2, x2, #0x160
    449c:	11404281 	add	w1, w20, #0x10, lsl #12
    44a0:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    44a4:	91062000 	add	x0, x0, #0x188
    44a8:	94000031 	bl	456c <spt_ComplainError>
    44ac:	b9400260 	ldr	w0, [x19]
    44b0:	11404000 	add	w0, w0, #0x10, lsl #12
    44b4:	1400001e 	b	452c <sptDumpMatrix+0x13c>
		for(sptIndex j=0; j < ncols; ++j) {
    44b8:	11000673 	add	w19, w19, #0x1
    44bc:	6b14027f 	cmp	w19, w20
    44c0:	54000400 	b.eq	4540 <sptDumpMatrix+0x150>  // b.none
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    44c4:	f9400ac0 	ldr	x0, [x22, #16]
    44c8:	bc735800 	ldr	s0, [x0, w19, uxtw #2]
    44cc:	1e22c000 	fcvt	d0, s0
    44d0:	aa1703e1 	mov	x1, x23
    44d4:	aa1503e0 	mov	x0, x21
    44d8:	97fff3c6 	bl	13f0 <fprintf@plt>
			spt_CheckOSError(iores < 0, "Mtx Dump");
    44dc:	36fffee0 	tbz	w0, #31, 44b8 <sptDumpMatrix+0xc8>
    44e0:	97fff3b8 	bl	13c0 <__errno_location@plt>
    44e4:	aa0003f8 	mov	x24, x0
    44e8:	b9400019 	ldr	w25, [x0]
    44ec:	3140433f 	cmn	w25, #0x10, lsl #12
    44f0:	54fffe40 	b.eq	44b8 <sptDumpMatrix+0xc8>  // b.none
    44f4:	2a1903e0 	mov	w0, w25
    44f8:	97fff372 	bl	12c0 <strerror@plt>
    44fc:	aa0003e4 	mov	x4, x0
    4500:	528005c3 	mov	w3, #0x2e                  	// #46
    4504:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7b8>
    4508:	91058042 	add	x2, x2, #0x160
    450c:	11404321 	add	w1, w25, #0x10, lsl #12
    4510:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7b8>
    4514:	91062000 	add	x0, x0, #0x188
    4518:	94000015 	bl	456c <spt_ComplainError>
    451c:	b9400300 	ldr	w0, [x24]
    4520:	11404000 	add	w0, w0, #0x10, lsl #12
    4524:	a94363f7 	ldp	x23, x24, [sp, #48]
    4528:	a9446bf9 	ldp	x25, x26, [sp, #64]
}
    452c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4530:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4534:	a94573fb 	ldp	x27, x28, [sp, #80]
    4538:	a8c77bfd 	ldp	x29, x30, [sp], #112
    453c:	d65f03c0 	ret
		iores = fprintf(fp, "\n");
    4540:	aa1503e1 	mov	x1, x21
    4544:	52800140 	mov	w0, #0xa                   	// #10
    4548:	97fff316 	bl	11a0 <fputc@plt>
	for(sptIndex i=0; i < nrows; ++i) {
    454c:	1100075a 	add	w26, w26, #0x1
    4550:	b9406fe0 	ldr	w0, [sp, #108]
    4554:	0b000294 	add	w20, w20, w0
    4558:	6b1a039f 	cmp	w28, w26
    455c:	54fff7c0 	b.eq	4454 <sptDumpMatrix+0x64>  // b.none
		for(sptIndex j=0; j < ncols; ++j) {
    4560:	4b1b0293 	sub	w19, w20, w27
    4564:	35fffb1b 	cbnz	w27, 44c4 <sptDumpMatrix+0xd4>
    4568:	17fffff6 	b	4540 <sptDumpMatrix+0x150>

000000000000456c <spt_ComplainError>:

/**
 * Set last error information as specified and print the error message.
 * Should not be called directly, use the macro `spt_CheckError`.
 */
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    456c:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    4570:	910003fd 	mov	x29, sp
    4574:	a90153f3 	stp	x19, x20, [sp, #16]
    4578:	a9025bf5 	stp	x21, x22, [sp, #32]
    457c:	aa0003f3 	mov	x19, x0
    4580:	2a0103f4 	mov	w20, w1
    4584:	aa0203f5 	mov	x21, x2
    4588:	2a0303f6 	mov	w22, w3
	g_last_error.errcode = errcode;
    458c:	d53bd046 	mrs	x6, tpidr_el0
    4590:	914000c6 	add	x6, x6, #0x0, lsl #12
    4594:	910040c6 	add	x6, x6, #0x10
    4598:	b90008c1 	str	w1, [x6, #8]
	g_last_error.module = module;
    459c:	f90000c0 	str	x0, [x6]
	g_last_error.file = file;
    45a0:	f90008c2 	str	x2, [x6, #16]
	g_last_error.line = line;
    45a4:	b90018c3 	str	w3, [x6, #24]
	if(reason) {
    45a8:	b40003a4 	cbz	x4, 461c <spt_ComplainError+0xb0>
    45ac:	a90363f7 	stp	x23, x24, [sp, #48]
    45b0:	aa0403f7 	mov	x23, x4
		free(g_last_error.reason);
    45b4:	aa0603f8 	mov	x24, x6
    45b8:	f94010c0 	ldr	x0, [x6, #32]
    45bc:	97fff365 	bl	1350 <free@plt>
		g_last_error.reason = strdup(reason);
    45c0:	aa1703e0 	mov	x0, x23
    45c4:	97fff337 	bl	12a0 <strdup@plt>
    45c8:	aa0003e6 	mov	x6, x0
    45cc:	f9001300 	str	x0, [x24, #32]
		if(!g_last_error.reason) {
    45d0:	b4000240 	cbz	x0, 4618 <spt_ComplainError+0xac>
    45d4:	a94363f7 	ldp	x23, x24, [sp, #48]
			abort();
		}
	}
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    45d8:	394000c0 	ldrb	w0, [x6]
    45dc:	350002c0 	cbnz	w0, 4634 <spt_ComplainError+0xc8>
						g_last_error.file,
						g_last_error.line,
						g_last_error.reason
		);
	} else {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    45e0:	2a1603e5 	mov	w5, w22
    45e4:	aa1503e4 	mov	x4, x21
    45e8:	2a1403e3 	mov	w3, w20
    45ec:	aa1303e2 	mov	x2, x19
    45f0:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    45f4:	91070021 	add	x1, x1, #0x1c0
    45f8:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    45fc:	f947d800 	ldr	x0, [x0, #4016]
    4600:	f9400000 	ldr	x0, [x0]
    4604:	97fff37b 	bl	13f0 <fprintf@plt>
						g_last_error.errcode,
						g_last_error.file,
						g_last_error.line
		);
	}
}
    4608:	a94153f3 	ldp	x19, x20, [sp, #16]
    460c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4610:	a8c47bfd 	ldp	x29, x30, [sp], #64
    4614:	d65f03c0 	ret
			abort();
    4618:	97fff33a 	bl	1300 <abort@plt>
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    461c:	d53bd040 	mrs	x0, tpidr_el0
    4620:	91400000 	add	x0, x0, #0x0, lsl #12
    4624:	91004000 	add	x0, x0, #0x10
    4628:	f9401006 	ldr	x6, [x0, #32]
    462c:	b5fffd66 	cbnz	x6, 45d8 <spt_ComplainError+0x6c>
    4630:	17ffffec 	b	45e0 <spt_ComplainError+0x74>
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4634:	2a1603e5 	mov	w5, w22
    4638:	aa1503e4 	mov	x4, x21
    463c:	2a1403e3 	mov	w3, w20
    4640:	aa1303e2 	mov	x2, x19
    4644:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    4648:	91068021 	add	x1, x1, #0x1a0
    464c:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    4650:	f947d800 	ldr	x0, [x0, #4016]
    4654:	f9400000 	ldr	x0, [x0]
    4658:	97fff366 	bl	13f0 <fprintf@plt>
    465c:	17ffffeb 	b	4608 <spt_ComplainError+0x9c>

0000000000004660 <sptBytesString>:
#include <stdlib.h>
#include "types.h"


char * sptBytesString(uint64_t const bytes)
{
    4660:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
    4664:	910003fd 	mov	x29, sp
    4668:	fd000be8 	str	d8, [sp, #16]
    466c:	9e670008 	fmov	d8, x0
	double size = (double)bytes;
    4670:	7e61d900 	ucvtf	d0, d8
	int suff = 0;
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4674:	f0000080 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    4678:	9109e000 	add	x0, x0, #0x278
    467c:	a9400c02 	ldp	x2, x3, [x0]
    4680:	a9028fe2 	stp	x2, x3, [sp, #40]
    4684:	a9410c02 	ldp	x2, x3, [x0, #16]
    4688:	a9038fe2 	stp	x2, x3, [sp, #56]
    468c:	f9401000 	ldr	x0, [x0, #32]
    4690:	f90027e0 	str	x0, [sp, #72]
	while(size > 1024 && suff < 5) {
    4694:	d2e81200 	mov	x0, #0x4090000000000000    	// #4652218415073722368
    4698:	9e670001 	fmov	d1, x0
    469c:	1e612010 	fcmpe	d0, d1
    46a0:	5400004c 	b.gt	46a8 <sptBytesString+0x48>
    46a4:	14000018 	b	4704 <sptBytesString+0xa4>
    46a8:	52800000 	mov	w0, #0x0                   	// #0
		size /= 1024.;
    46ac:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    46b0:	fd411c22 	ldr	d2, [x1, #568]
	while(size > 1024 && suff < 5) {
    46b4:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    46b8:	fd412021 	ldr	d1, [x1, #576]
		size /= 1024.;
    46bc:	1e620800 	fmul	d0, d0, d2
		++suff;
    46c0:	11000400 	add	w0, w0, #0x1
	while(size > 1024 && suff < 5) {
    46c4:	7100101f 	cmp	w0, #0x4
    46c8:	1e61d414 	fccmpe	d0, d1, #0x4, le
    46cc:	54ffff8c 	b.gt	46bc <sptBytesString+0x5c>
	}
	char * ret = NULL;
    46d0:	f90013ff 	str	xzr, [sp, #32]
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    46d4:	9100a3e1 	add	x1, sp, #0x28
    46d8:	f860d822 	ldr	x2, [x1, w0, sxtw #3]
    46dc:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    46e0:	91082021 	add	x1, x1, #0x208
    46e4:	910083e0 	add	x0, sp, #0x20
    46e8:	97fff2b6 	bl	11c0 <asprintf@plt>
    46ec:	3100041f 	cmn	w0, #0x1
    46f0:	540000e0 	b.eq	470c <sptBytesString+0xac>  // b.none
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
		ret = NULL;
	}
	return ret;
}
    46f4:	f94013e0 	ldr	x0, [sp, #32]
    46f8:	fd400be8 	ldr	d8, [sp, #16]
    46fc:	a8c57bfd 	ldp	x29, x30, [sp], #80
    4700:	d65f03c0 	ret
	int suff = 0;
    4704:	52800000 	mov	w0, #0x0                   	// #0
    4708:	17fffff2 	b	46d0 <sptBytesString+0x70>
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
    470c:	9e660102 	fmov	x2, d8
    4710:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7b8>
    4714:	91084021 	add	x1, x1, #0x210
    4718:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x100e8>
    471c:	f947d800 	ldr	x0, [x0, #4016]
    4720:	f9400000 	ldr	x0, [x0]
    4724:	97fff333 	bl	13f0 <fprintf@plt>
		ret = NULL;
    4728:	f90013ff 	str	xzr, [sp, #32]
    472c:	17fffff2 	b	46f4 <sptBytesString+0x94>

0000000000004730 <sptRandomValue>:


sptValue sptRandomValue(void)
{
    4730:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4734:	910003fd 	mov	x29, sp
    4738:	fd000be8 	str	d8, [sp, #16]
	sptValue v =  3.0 * ((sptValue) rand() / (sptValue) RAND_MAX);
    473c:	97fff2d5 	bl	1290 <rand@plt>
    4740:	1e028408 	scvtf	s8, w0, #31
    4744:	1e211000 	fmov	s0, #3.000000000000000000e+00
    4748:	1e200908 	fmul	s8, s8, s0
	if(rand() % 2 == 0) {
    474c:	97fff2d1 	bl	1290 <rand@plt>
		v *= -1;
    4750:	1e214100 	fneg	s0, s8
    4754:	f240001f 	tst	x0, #0x1
	}
	return v;
}
    4758:	1e280c00 	fcsel	s0, s0, s8, eq  // eq = none
    475c:	fd400be8 	ldr	d8, [sp, #16]
    4760:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4764:	d65f03c0 	ret
    4768:	d503201f 	nop
    476c:	d503201f 	nop

0000000000004770 <__aarch64_cas4_acq_rel>:
    4770:	d503245f 	bti	c
    4774:	f0000090 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    4778:	394a8610 	ldrb	w16, [x16, #673]
    477c:	34000070 	cbz	w16, 4788 <__aarch64_cas4_acq_rel+0x18>
    4780:	88e0fc41 	casal	w0, w1, [x2]
    4784:	d65f03c0 	ret
    4788:	2a0003f0 	mov	w16, w0
    478c:	885ffc40 	ldaxr	w0, [x2]
    4790:	6b10001f 	cmp	w0, w16
    4794:	54000061 	b.ne	47a0 <__aarch64_cas4_acq_rel+0x30>  // b.any
    4798:	8811fc41 	stlxr	w17, w1, [x2]
    479c:	35ffff91 	cbnz	w17, 478c <__aarch64_cas4_acq_rel+0x1c>
    47a0:	d65f03c0 	ret
    47a4:	d503201f 	nop
    47a8:	d503201f 	nop
    47ac:	d503201f 	nop

00000000000047b0 <__libc_csu_init>:
    47b0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    47b4:	910003fd 	mov	x29, sp
    47b8:	a90153f3 	stp	x19, x20, [sp, #16]
    47bc:	d0000094 	adrp	x20, 16000 <__FRAME_END__+0x100e8>
    47c0:	9134e294 	add	x20, x20, #0xd38
    47c4:	a9025bf5 	stp	x21, x22, [sp, #32]
    47c8:	d0000095 	adrp	x21, 16000 <__FRAME_END__+0x100e8>
    47cc:	9134a2b5 	add	x21, x21, #0xd28
    47d0:	cb150294 	sub	x20, x20, x21
    47d4:	2a0003f6 	mov	w22, w0
    47d8:	a90363f7 	stp	x23, x24, [sp, #48]
    47dc:	aa0103f7 	mov	x23, x1
    47e0:	aa0203f8 	mov	x24, x2
    47e4:	9343fe94 	asr	x20, x20, #3
    47e8:	97fff254 	bl	1138 <_init>
    47ec:	b4000174 	cbz	x20, 4818 <__libc_csu_init+0x68>
    47f0:	d2800013 	mov	x19, #0x0                   	// #0
    47f4:	d503201f 	nop
    47f8:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
    47fc:	aa1803e2 	mov	x2, x24
    4800:	91000673 	add	x19, x19, #0x1
    4804:	aa1703e1 	mov	x1, x23
    4808:	2a1603e0 	mov	w0, w22
    480c:	d63f0060 	blr	x3
    4810:	eb13029f 	cmp	x20, x19
    4814:	54ffff21 	b.ne	47f8 <__libc_csu_init+0x48>  // b.any
    4818:	a94153f3 	ldp	x19, x20, [sp, #16]
    481c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4820:	a94363f7 	ldp	x23, x24, [sp, #48]
    4824:	a8c47bfd 	ldp	x29, x30, [sp], #64
    4828:	d65f03c0 	ret
    482c:	d503201f 	nop

0000000000004830 <__libc_csu_fini>:
    4830:	d65f03c0 	ret

Disassembly of section .fini:

0000000000004834 <_fini>:
    4834:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    4838:	910003fd 	mov	x29, sp
    483c:	a8c17bfd 	ldp	x29, x30, [sp], #16
    4840:	d65f03c0 	ret
