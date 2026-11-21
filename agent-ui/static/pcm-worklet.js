class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.input = [];
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    this.input.push(...channel);
    const inputFrames = Math.round(sampleRate * 0.05);

    while (this.input.length >= inputFrames) {
      const source = this.input.splice(0, inputFrames);
      const output = new Int16Array(800);

      for (let i = 0; i < output.length; i += 1) {
        const start = Math.floor(i * source.length / output.length);
        const end = Math.max(
          start + 1,
          Math.floor((i + 1) * source.length / output.length),
        );
        let sum = 0;
        for (let j = start; j < end; j += 1) sum += source[j];
        const value = Math.max(-1, Math.min(1, sum / (end - start)));
        output[i] = value < 0 ? value * 32768 : value * 32767;
      }

      this.port.postMessage(output.buffer, [output.buffer]);
    }

    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
