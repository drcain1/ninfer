/* Match the CUDA section order from `nvcc -hls gen-lcs -shared -aug-hls`,
 * augmenting the host's default script instead of copying its platform layout.
 * Registration records use absolute pointers to the large device payloads.
 */
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  __nv_relfatbin : { *(__nv_relfatbin) }
  .nv_fatbin : { *(.nv_fatbin) }
}
INSERT AFTER .bss;
