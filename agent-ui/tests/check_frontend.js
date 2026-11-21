const fs = require("node:fs");
const vm = require("node:vm");

const page = fs.readFileSync("agent-ui/static/index.html", "utf8");
const script = page.match(/<script>([\s\S]*)<\/script>/);
if (!script) throw new Error("Missing browser application script.");
new vm.Script(script[1], { filename: "index.html" });

const workletSource = fs.readFileSync(
  "agent-ui/static/pcm-worklet.js",
  "utf8",
);
let Worklet;
const posted = [];
class AudioWorkletProcessor {
  constructor() {
    this.port = {
      postMessage(data) {
        posted.push(data);
      },
    };
  }
}

vm.runInNewContext(workletSource, {
  AudioWorkletProcessor,
  Int16Array,
  Math,
  registerProcessor(name, processor) {
    if (name !== "pcm-worklet") {
      throw new Error(`Unexpected worklet name: ${name}`);
    }
    Worklet = processor;
  },
  sampleRate: 48_000,
});

if (!Worklet) throw new Error("PCM worklet was not registered.");
const worklet = new Worklet();
worklet.process([[new Float32Array(2_400).fill(0.5)]]);
if (posted.length !== 1) {
  throw new Error(`Expected one 50 ms audio frame, got ${posted.length}.`);
}
if (posted[0].byteLength !== 1_600) {
  throw new Error(`Expected 1,600 PCM bytes, got ${posted[0].byteLength}.`);
}
console.log("Browser script syntax and PCM frame shape OK.");
