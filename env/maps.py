# Cleanup colors
# '@' means "wall"
# 'H' is potential waste spawn point，有污染的河流
# 'R' is river cell，干净的河流但可能产生污染
# 'S' is stream cell，干净的河流且不可能产生污染
# 'P' means "player" spawn point
# 'A' means apple spawn point，实实在在有的苹果
# 'B' is potential apple spawn point，orchard，苹果可以从这长出来，但是现在没苹果
# ' ' is empty space

CLEANUP_10x10_SYM = [
    '@@@@@@@@@@',
    '@HH   P B@',
    '@RR    BB@',
    '@HH     B@',
    '@RR    BB@',
    '@HH P   B@',
    '@RR    BB@',
    '@HH     B@',
    '@RRP   BB@',
    '@@@@@@@@@@']

# 7x7 map: Agent 0 on river side, Agent 1 on apple side
# 5个orchard
CLEANUP_SMALL_SYM = [
    '@@@@@@@',
    '@H  PB@',
    '@H   B@',
    '@    B@',
    '@    B@',
    '@ P  B@',
    '@@@@@@@']

CLEANUP_SMALL_same_col = [
    '@@@@@@@',
    '@H   B@',
    '@H   B@',
    '@ P  B@',
    '@    B@',
    '@ P  B@',
    '@@@@@@@']

# 21个orchard
CLEANUP_7x7_allapple = [
    '@@@@@@@',
    '@HBBPB@',
    '@HBBBB@',
    '@BBBBB@',
    '@BBBBB@',
    '@BPBBB@',
    '@@@@@@@']

CLEANUP_SMALL_double = [
    '@@@@@@@',
    '@HHPBB@',
    '@HH BB@',
    '@   BB@',
    '@   BB@',
    '@ P BB@',
    '@@@@@@@']

CLEANUP_SMALL_rightup = [
    '@@@@@@@',
    '@H  PB@',
    '@H   B@',
    '@    B@',
    '@    B@',
    '@    B@',
    '@@@@@@@']

CLEANUP_SMALL_new = [
    '@@@@@@@',
    '@BBBBB@',
    '@     @',
    '@     @',
    '@     @',
    '@P H P@',
    '@@@@@@@']

CLEANUP_easy = [
    '@@@@@',
    '@B B@',
    '@   @',
    '@PHP@',
    '@@@@@']

CLEANUP_11x11_allapple = [
    '@@@@@@@@@@@',
    '@RRBBBBBRR@',
    '@HHBBBBBHH@',
    '@RRBBBBBRR@',
    '@HHBBBBBHH@',
    '@RRBBBBBRR@',
    '@HHBBBBBHH@',
    '@RRBBBBBRR@',
    '@HHBBBBBHH@',
    '@RRP   PRR@',
    '@@@@@@@@@@@']

CLEANUP_11x11_rightwaste = [
    '@@@@@@@@@@@',
    '@BBBBBBBRR@',
    '@BBBBBBBHH@',
    '@BBBBBBBRR@',
    '@BBBBBBPHH@',
    '@BBBBBBBRR@',
    '@BBBBBBBHH@',
    '@BBBBBBBRR@',
    '@BBBBBBBHH@',
    '@BBBBBBPRR@',
    '@@@@@@@@@@@']

CLEANUP_11x11_origin = [
    '@@@@@@@@@@@',
    '@RR    PBB@',
    '@HH      B@',
    '@RR     BB@',
    '@HH      B@',
    '@RR P   BB@',
    '@HH      B@',
    '@RR     BB@',
    '@HH      B@',
    '@RRP    BB@',
    '@@@@@@@@@@@']

# 25*18
CLEANUP_MAP = [
    '   @@@@@@@@@@@@@@@@@@    ',
    '   @RRRRRR     BBBBB@    ',
    '   @HHHHHH      BBBB@    ',
    '   @RRRRRR     BBBBB@    ',
    '   @RRRRR  P    BBBB@    ',
    '   @RRRRR    P BBBBB@    ',
    '   @HHHHH       BBBB@    ',
    '   @RRRRR      BBBBB@    ',
    '   @HHHHHHSSSSSSBBBB@    ',
    '   @HHHHHHSSSSSSBBBB@    ',
    '   @RRRRR   P P BBBB@    ',
    '   @HHHHH   P  BBBBB@    ',
    '   @RRRRRR    P BBBB@    ',
    '   @HHHHHH P   BBBBB@    ',
    '   @RRRRR       BBBB@    ',
    '   @HHHH    P  BBBBB@    ',
    '   @RRRRR       BBBB@    ',
    '   @HHHHH  P P BBBBB@    ',
    '   @RRRRR       BBBB@    ',
    '   @HHHH       BBBBB@    ',
    '   @RRRRR       BBBB@    ',
    '   @HHHHH      BBBBB@    ',
    '   @RRRRR       BBBB@    ',
    '   @HHHH       BBBBB@    ',
    '   @@@@@@@@@@@@@@@@@@    ']

# 21行31列
CLEANUP_big = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@HRRRHRRHRHRHRHRHRHRHHRHRRRHR@',
    '@HRHRHRRHRHRHRHRHRHRHHRHRRRHR@',
    '@HRRHRRHHRHRHRHRHRHRHHRHRRRHR@',
    '@HRHRHRRHRHRHRHRHRHRHHRHRRRHR@',
    '@HRRRRRRHRHRHRHRHRHRHHRHRRRHR@',
    '@               HRHHHHHH     @',
    '@   P    P          SSS      @',
    '@     P     P   P   SS   P   @',
    '@             P   PPSS       @',
    '@   P    P          SS    P  @',
    '@               P   SS P     @',
    '@     P           P SS       @',
    '@           P       SS  P    @',
    '@  P             P PSS       @',
    '@ B B B B B B B B B SSB B B B@',
    '@BBBBBBBBBBBBBBBBBBBBBBBBBBBB@',
    '@BBBBBBBBBBBBBBBBBBBBBBBBBBBB@',
    '@BBBBBBBBBBBBBBBBBBBBBBBBBBBB@',
    '@BBBBBBBBBBBBBBBBBBBBBBBBBBBB@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
]
'''
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@HRRRHRRHRHRHRHRHRHRHHRHRRRHR@@',
    '@HRHRHRRHRHRHRHRHRHRHHRHRRRHR@@',
    '@HRRHRRHHRHRHRHRHRHRHHRHRRRHR@@',
    '@HRHRHRRHRHRHRHRHRHRHHRHRRRHR@@',
    '@HRRRRRRHRHRHRHRHRHRHHRHRRRHR@@',
    '@               HRHHHHHH     @@',
    '@   P    P          SSS      @@',
    '@     P     P   P   SS   P   @@',
    '@             P   PPSS       @@',
    '@   P    P          SS    P  @@',
    '@               P   SS P     @@',
    '@     P           P SS       @@',
    '@           P       SS  P    @@',
    '@  P             P PSS       @@',
    '@ B B B B B B B B B SSB B B B@@',
    '@BBBBBBBBBBBBBBBBBBBBBBBBBBBB@@',
    '@BBBBBBBBBBBBBBBBBBBBBBBBBBBB@@',
    '@BBBBBBBBBBBBBBBBBBBBBBBBBBBB@@',
    '@BBBBBBBBBBBBBBBBBBBBBBBBBBBB@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
'''

HARVEST_MAP_origin = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

HARVEST_MAP_11 = [
    '@@@@@@@@@@@',
    '@ P    A  @',
    '@     AAA @',
    '@    AAAAA@',
    '@     AAA @',
    '@  A   A  @',
    '@ AAA     @',
    '@AAAAA    @',
    '@ AAA     @',
    '@  A    P @',
    '@@@@@@@@@@@',
]

HARVEST_MAP_7 = [
    '@@@@@@@',
    '@P A  @',
    '@ AAA @',
    '@AAAAA@',
    '@ AAA @',
    '@  A P@',
    '@@@@@@@',
]
