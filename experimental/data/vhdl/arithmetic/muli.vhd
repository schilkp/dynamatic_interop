library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity muli is
  generic (
    BITWIDTH : integer
  );
  port (
    clk         : in std_logic;
    rst         : in std_logic;
    pValidArray : in std_logic_vector(1 downto 0);
    nReady      : in std_logic;
    valid       : out std_logic;
    readyArray  : out std_logic_vector(1 downto 0);
    --dataInArray
    inToShift    : in std_logic_vector(BITWIDTH - 1 downto 0);
    inShiftBy    : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0));
end entity;

architecture arch of muli is

  signal join_valid : std_logic;

  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic;

  -- multiplier latency (4 or 8)
  constant LATENCY : integer := 4;
  --constant LATENCY : integer := 8;

begin
  join : entity work.join(arch) generic map(2)
    port map(
      pValidArray,
      oehb_ready,
      join_valid,
      readyArray);

  -- instantiated multiplier (work.mul_4_stage or work.mul_8_stage)
  multiply_unit : entity work.mul_4_stage(behav) generic map (BITWIDTH)
    --multiply_unit:  entity work.mul_8_stage(behav) generic map (BITWIDTH)
    port map(
      clk => clk,
      ce  => oehb_ready,
      a   => inToShift,
      b   => inShiftBy,
      p   => dataOutArray);

  buff : entity work.delay_buffer(arch) generic map(LATENCY - 1)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid);

  oehb : entity work.OEHB(arch) generic map (1)
    port map(
      --inputspValidArray
      clk            => clk,
      rst            => rst,
      pValidArray(0) => buff_valid, -- real or speculatef condition (determined by merge1)
      nReady         => nReady,
      valid          => valid,
      --outputs
      readyArray(0) => oehb_ready,
      inToShift     => oehb_datain,
      dataOutArray  => oehb_dataOut);
end architecture;
