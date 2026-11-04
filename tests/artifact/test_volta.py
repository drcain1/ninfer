"""Lossless Volta layout addressing, independently of CUDA numerical math."""
import unittest
from tools.artifact.codecs.volta import pack_planes, unpack_planes
from tools.artifact.layouts import encoded_size


class VoltaLayoutTest(unittest.TestCase):
    def test_all_code_words_round_trip(self):
        for fmt in ("nvfp4", "fp8_e4m3fn_row_bf16"):
            n, k = 256, 256
            codes = bytes((i*7+13)%256 for i in range(n*k//(2 if fmt == "nvfp4" else 1)))
            scales = bytes((i*17+31)%256 for i in range(n*k//16 if fmt == "nvfp4" else n*2))
            packed = pack_planes(codes, scales, fmt, (n,k))
            self.assertEqual(unpack_planes(*packed,fmt,(n,k)),(codes,scales))

    def test_fragment_coordinates(self):
        n,k=128,128
        codes=bytes([0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe])*(n*k//16)
        scales=bytes(i%127 for i in range(n*k//16))
        packed, scale=pack_planes(codes,scales,"nvfp4",(n,k))
        # Logical row 5 is physical lane 17; adjacent half2 operands consume
        # nibble pairs (0,1), (2,3), ... after the shift decoder.
        self.assertEqual(packed[17*8:18*8],bytes([0x20,0x64,0x31,0x75,0xa8,0xec,0xb9,0xfd]))
        self.assertEqual(scale[17],scales[5*8])
        codes=bytes(range(128))*128
        packed,scale=pack_planes(codes,bytes(n*2),"fp8_e4m3fn_row_bf16",(n,k))
        self.assertEqual(packed[17*16:18*16],bytes(range(16)))

    def test_reject_incomplete_tiles(self):
        for shape in ((127,128),(128,129),(128,0)):
            with self.assertRaises(ValueError):
                encoded_size("volta_qpn_v1","nvfp4",shape)
        with self.assertRaises(ValueError):
            encoded_size("volta_qpn_v1","q8_g32_fp16",(128,128))

if __name__ == "__main__":
    unittest.main()
