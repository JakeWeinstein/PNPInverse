// Funder/electrochemist deck — outline + restored original technical slides.
// Strong form (orig s7), all-BCs domain figure (orig s8), numerically-hard
// (orig s9) and how-we-solve-it (orig s10) verbatim, continuation (orig s11).
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const BASE = "/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse";
const FIGS = BASE + "/writeups/GroupMeetingTalk/figures";
const PRES = BASE + "/writeups/5:27Presentation";
const ASSETS = PRES + "/build/assets";
fs.mkdirSync(ASSETS, { recursive: true });
fs.copyFileSync("/tmp/may27_media/ppt/media/image-7-1.png", ASSETS + "/exp_data.png");

const IMG = {
  domain: { p: FIGS + "/s8_s10_domain_double_layer.png", r: 2.55 },
  cont:   { p: FIGS + "/continuation_bisection.png",     r: 2.55 },
  exp:    { p: ASSETS + "/exp_data.png",                 r: 0.92 },
  jithin: { p: BASE + "/StudyResults/jithin_fig_4_26_4_27_4_28/compare_all.png", r: 0.93 },
  mms:    { p: FIGS + "/s22_mms_rates.png",              r: 1.48 },
  eq1: { p: ASSETS + "/eq1.png", r: 9.54 }, eq2: { p: ASSETS + "/eq2.png", r: 13.02 },
  eq3: { p: ASSETS + "/eq3.png", r: 5.88 }, eq4: { p: ASSETS + "/eq4.png", r: 12.73 },
  eq5: { p: ASSETS + "/eq5.png", r: 7.53 }, eq6: { p: ASSETS + "/eq6.png", r: 5.43 },
  eq7: { p: ASSETS + "/eq7.png", r: 12.49 }, eq8: { p: ASSETS + "/eq8.png", r: 20.02 },
  eq9: { p: ASSETS + "/eq9.png", r: 9.13 },
};

const NAVY = "16243B", INK = "1B2A4A", BLUE = "2E6BE6", ICE = "C7D6F2", BODY = "3B4658",
      MUTE = "8A94A6", WHITE = "FFFFFF", LINEC = "E2E8F2", CARD = "F4F7FC",
      GREEN = "2E9E6B", LOSS = "C0544B", AMBER = "B06A1E";
const HEAD = "Trebuchet MS", FONT = "Calibri", CODE = "Consolas", CODEBG = "0F1B2D", CODETX = "DCE6F5", CODECOM = "7486A2";

const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 10, height: 5.625 });
pres.layout = "W";
pres.author = "Jake Weinstein";
pres.title = "A general solver for electrochemical transport–reaction systems";
const R = pres.shapes.RECTANGLE, RR = pres.shapes.ROUNDED_RECTANGLE, OV = pres.shapes.OVAL, LN = pres.shapes.LINE;
const mkShadow = () => ({ type: "outer", color: "000000", blur: 9, offset: 3, angle: 135, opacity: 0.22 });

function imgFit(slide, im, box) {
  let w = box.w, h = w / im.r;
  if (h > box.h) { h = box.h; w = h * im.r; }
  const x = box.x + (box.w - w) / 2, y = box.y + (box.h - h) / 2;
  slide.addImage({ path: im.p, x, y, w, h }); return { x, y, w, h };
}
function eqImg(slide, im, y, maxH, cx) {
  const w = maxH * im.r, h = maxH, x = cx !== undefined ? cx : (10 - w) / 2;
  slide.addImage({ path: im.p, x, y, w, h }); return { x, y, w, h };
}
function footer(slide, n) {
  slide.addText("PNP–BV solver", { x: 0.5, y: 5.30, w: 4, h: 0.28, fontSize: 9, color: MUTE, fontFace: FONT, valign: "middle", margin: 0 });
  slide.addText(String(n), { x: 9.0, y: 5.30, w: 0.5, h: 0.28, fontSize: 9, color: MUTE, fontFace: FONT, align: "right", valign: "middle", margin: 0 });
}
function head(slide, section, title) {
  slide.addText(section.toUpperCase(), { x: 0.78, y: 0.30, w: 8.9, h: 0.25, fontSize: 11, bold: true, color: BLUE, charSpacing: 2.5, fontFace: FONT, margin: 0 });
  slide.addShape(R, { x: 0.5, y: 0.70, w: 0.17, h: 0.17, fill: { color: BLUE } });
  slide.addText(title, { x: 0.78, y: 0.56, w: 8.9, h: 0.55, fontSize: 28, bold: true, color: INK, fontFace: HEAD, valign: "middle", margin: 0 });
}
function chip(slide, x, y, w, label, col) {
  slide.addShape(RR, { x, y, w, h: 0.5, fill: { color: WHITE }, line: { color: col, width: 1.25 }, rectRadius: 0.25 });
  slide.addText(label, { x: x + 0.06, y, w: w - 0.12, h: 0.5, fontSize: 11.5, bold: true, color: col, align: "center", valign: "middle", fontFace: FONT, margin: 0 });
}
function badge(slide, x, y, n) {
  slide.addShape(RR, { x, y, w: 0.4, h: 0.4, fill: { color: NAVY }, rectRadius: 0.05 });
  slide.addText(String(n), { x, y, w: 0.4, h: 0.4, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle", fontFace: HEAD, margin: 0 });
}
let s;

// ===================================================== 1 · TITLE
s = pres.addSlide(); s.background = { color: NAVY };
s.addShape(R, { x: 0, y: 0, w: 0.16, h: 5.625, fill: { color: BLUE } });
s.addShape(R, { x: 0.9, y: 1.55, w: 0.42, h: 0.42, fill: { color: BLUE } });
s.addText("A general solver for electrochemical\ntransport–reaction systems", {
  x: 0.86, y: 2.1, w: 8.7, h: 1.3, fontSize: 33, bold: true, color: WHITE, fontFace: HEAD, lineSpacingMultiple: 1.06, margin: 0 });
s.addText("Declare the system; the solver handles the math — demonstrated on ORR selectivity", {
  x: 0.9, y: 3.66, w: 8.5, h: 0.5, fontSize: 16.5, color: ICE, fontFace: FONT, margin: 0 });
s.addShape(R, { x: 0.92, y: 4.55, w: 1.6, h: 0.028, fill: { color: BLUE } });
s.addText("Jake Weinstein", { x: 0.9, y: 4.7, w: 6, h: 0.3, fontSize: 13, bold: true, color: "9FB2D6", fontFace: FONT, margin: 0 });

// ===================================================== 2 · CLASS OF PROBLEM
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "The class of problem", "One framework, many systems");
s.addText("Migration–diffusion–reaction with a structured double layer — the same mathematics across electrochemistry.",
  { x: 0.78, y: 1.12, w: 8.9, h: 0.35, fontSize: 13.5, color: BODY, fontFace: FONT, margin: 0 });
["ORR", "CO₂RR", "Electrodeposition", "Corrosion"].forEach((t, i) => chip(s, 0.78 + i * 2.18, 1.62, 2.0, t, BLUE));
const cy = 2.9, ch = 1.4;
s.addText("Stern", { x: 1.0, y: cy - 0.34, w: 1.5, h: 0.25, fontSize: 10.5, bold: true, color: AMBER, align: "left", fontFace: FONT, margin: 0 });
s.addShape(R, { x: 1.0, y: cy, w: 0.32, h: ch, fill: { color: NAVY } });
s.addShape(R, { x: 1.32, y: cy, w: 0.2, h: ch, fill: { color: AMBER } });
s.addShape(R, { x: 1.52, y: cy, w: 2.72, h: ch, fill: { color: "EAF1FE" } });
s.addShape(R, { x: 4.24, y: cy, w: 4.0, h: ch, fill: { color: WHITE }, line: { color: LINEC, width: 1 } });
s.addText("Electrode", { x: 0.7, y: cy + ch + 0.06, w: 1.0, h: 0.25, fontSize: 10.5, color: BODY, align: "center", fontFace: FONT, margin: 0 });
s.addText("Diffuse layer", { x: 1.9, y: cy + ch + 0.06, w: 2.0, h: 0.25, fontSize: 10.5, color: BLUE, align: "center", fontFace: FONT, margin: 0 });
s.addText("Bulk", { x: 5.6, y: cy + ch + 0.06, w: 1.3, h: 0.25, fontSize: 10.5, color: MUTE, align: "center", fontFace: FONT, margin: 0 });
[[1.72, BLUE], [2.0, LOSS], [2.35, BLUE], [2.68, LOSS], [2.5, BLUE], [3.12, BLUE], [3.5, LOSS], [3.95, BLUE], [4.8, LOSS], [5.7, BLUE], [6.9, LOSS]]
  .forEach(([ix, col], k) => s.addShape(OV, { x: ix, y: cy + 0.22 + (k % 4) * 0.27, w: 0.13, h: 0.13, fill: { color: col } }));
s.addText("We'll demonstrate the workflow on ORR.", { x: 0.78, y: 4.78, w: 8, h: 0.3, fontSize: 13, italic: true, color: MUTE, fontFace: FONT, margin: 0 });
footer(s, 2);

// ===================================================== 3 · THESIS
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "The idea", "Two halves of the problem");
function tcard(x, title, sub, items, accent) {
  s.addShape(RR, { x, y: 1.45, w: 4.05, h: 3.45, fill: { color: CARD }, line: { color: LINEC, width: 1 }, rectRadius: 0.08, shadow: mkShadow() });
  s.addShape(R, { x, y: 1.45, w: 4.05, h: 0.62, fill: { color: accent } });
  s.addText(title, { x: x + 0.25, y: 1.45, w: 3.6, h: 0.62, fontSize: 18, bold: true, color: WHITE, fontFace: HEAD, valign: "middle", margin: 0 });
  s.addText(sub, { x: x + 0.25, y: 2.16, w: 3.6, h: 0.3, fontSize: 11.5, italic: true, color: MUTE, fontFace: FONT, margin: 0 });
  s.addText(items.map((t) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: true, color: BODY, fontSize: 14.5 } })),
    { x: x + 0.28, y: 2.55, w: 3.55, h: 2.2, fontFace: FONT, margin: 0, paraSpaceAfter: 8 });
}
tcard(0.55, "Model specification", "the electrochemistry", ["Species (dynamic & analytic)", "Reactions & stoichiometry", "Extra physics terms", "Boundary conditions"], INK);
tcard(5.40, "Numerical machinery", "the solver's job", ["Formulation & log-space", "Meshing (1D / 2D)", "Continuation & recovery", "Robust convergence"], BLUE);
s.addShape(OV, { x: 4.62, y: 2.85, w: 0.76, h: 0.76, fill: { color: WHITE }, line: { color: BLUE, width: 1.5 }, shadow: mkShadow() });
s.addText("→", { x: 4.62, y: 2.82, w: 0.76, h: 0.76, fontSize: 26, bold: true, color: BLUE, align: "center", valign: "middle", fontFace: HEAD, margin: 0 });
footer(s, 3);

// ===================================================== 4 · WHAT YOU SPECIFY 1 — strong form
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "Governing equations & terms · menu 1", "What you specify");
function modelRow(yT, lead, desc, im, eqH) {
  s.addText([{ text: lead + "  ", options: { bold: true, color: INK } }, { text: desc, options: { color: BODY } }],
    { x: 0.7, y: yT, w: 8.9, h: 0.3, fontSize: 13.5, fontFace: FONT, margin: 0 });
  eqImg(s, im, yT + 0.33, eqH);
}
modelRow(1.46, "PNP in the diffuse layer —", "ions diffuse and electromigrate in the field.", IMG.eq1, 0.46);
modelRow(2.42, "Butler–Volmer flux —", "rate scales with exp(overpotential), tied to the applied voltage.", IMG.eq2, 0.32);
modelRow(3.26, "Steric correction —", "removes non-physical ion concentrations.", IMG.eq3, 0.48);
modelRow(4.22, "Stern layer —", "enters as a Robin boundary condition.", IMG.eq4, 0.3);
footer(s, 4);

// ===================================================== 5 · WHAT YOU SPECIFY 2 — boundary conditions
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "Boundary conditions · menu 2", "What you specify");
const bcs = [["Bulk Dirichlet", GREEN], ["Butler–Volmer flux", BLUE], ["Stern Robin", AMBER], ["No-flux", MUTE], ["Analytic Bikerman", INK]];
bcs.forEach((b, i) => chip(s, 0.5 + i * 1.86, 1.28, 1.74, b[0], b[1]));
imgFit(s, IMG.domain, { x: 0.4, y: 2.0, w: 9.2, h: 3.0 });
s.addText("Mix and match per species, per boundary.", { x: 0.5, y: 5.0, w: 9, h: 0.28, fontSize: 12, italic: true, color: MUTE, align: "center", fontFace: FONT, margin: 0 });
footer(s, 5);

// ===================================================== 6 · WHAT YOU SPECIFY 3 — reactions
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "Reactions & stoichiometry · menu 3", "What you specify");
function rxnCard(x, eq, meta, col) {
  s.addShape(RR, { x, y: 1.5, w: 4.05, h: 1.5, fill: { color: CARD }, line: { color: col, width: 1.25 }, rectRadius: 0.08 });
  s.addShape(R, { x, y: 1.5, w: 0.1, h: 1.5, fill: { color: col } });
  s.addText(eq, { x: x + 0.3, y: 1.65, w: 3.6, h: 0.7, fontSize: 18, bold: true, color: INK, fontFace: HEAD, valign: "middle", margin: 0 });
  s.addText(meta, { x: x + 0.3, y: 2.35, w: 3.6, h: 0.5, fontSize: 13, color: col, bold: true, fontFace: FONT, margin: 0 });
}
rxnCard(0.55, "O₂ + 2 H⁺ + 2 e⁻ → H₂O₂", "E° = 0.695 V   ·   2e⁻ — the target", GREEN);
rxnCard(5.40, "O₂ + 4 H⁺ + 4 e⁻ → 2 H₂O", "E° = 1.23 V   ·   4e⁻ — the loss", LOSS);
s.addText("Declare stoichiometry · nₑ · E° · α · k₀   →   log-rate Butler–Volmer   (parallel or sequential)",
  { x: 0.5, y: 3.5, w: 9, h: 0.35, fontSize: 13.5, color: BODY, align: "center", fontFace: FONT, margin: 0 });
s.addShape(RR, { x: 1.7, y: 4.15, w: 6.6, h: 0.62, fill: { color: NAVY }, rectRadius: 0.08 });
s.addText("Edit a coefficient or append a reaction — the flux re-assembles.",
  { x: 1.7, y: 4.15, w: 6.6, h: 0.62, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", fontFace: FONT, margin: 0 });
footer(s, 6);

// ===================================================== 7 · WHY NUMERICALLY HARD (orig s9)
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "The difficulty", "Why it's numerically hard");
s.addText([
  { text: "Concentrations span ~10 orders of magnitude", options: { bullet: { indent: 14 }, bold: true, color: INK } },
  { text: " across the domain.", options: { color: BODY, breakLine: true } },
  { text: "The potential varies sharply", options: { bullet: { indent: 14 }, bold: true, color: INK } },
  { text: " right at the Stern layer.", options: { color: BODY, breakLine: true } },
  { text: "The exponential Butler–Volmer term", options: { bullet: { indent: 14 }, bold: true, color: INK } },
  { text: " drives steep buildup in c and φ at the boundary.", options: { color: BODY, breakLine: true } },
  { text: "Diffusion and electromigration", options: { bullet: { indent: 14 }, bold: true, color: INK } },
  { text: " act on very different timescales — the system is stiff.", options: { color: BODY, breakLine: true } },
], { x: 0.6, y: 1.5, w: 9.0, h: 1.7, fontSize: 14.5, fontFace: FONT, margin: 0, paraSpaceAfter: 9 });
eqImg(s, IMG.eq5, 3.45, 0.55);
eqImg(s, IMG.eq6, 4.45, 0.52);
footer(s, 7);

// ===================================================== 8 · HOW WE SOLVE IT (orig s10)
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "The solver", "How we solve it");
function solveRow(yT, n, lead, desc, im, eqH) {
  badge(s, 0.6, yT, n);
  s.addText([{ text: lead, options: { bold: true, color: INK } }, { text: "  " + desc, options: { color: BODY } }],
    { x: 1.18, y: yT + 0.02, w: 8.3, h: 0.35, fontSize: 14, fontFace: FONT, margin: 0, valign: "middle" });
  if (im) eqImg(s, im, yT + 0.36, eqH, 1.18);
}
solveRow(1.45, 1, "Solve μ = ln c + z·φ", "— mirrors the no-flux analytic solution.", IMG.eq7, 0.34);
solveRow(2.4, 2, "Log-rate Butler–Volmer", "— build the rate in log-space, exponentiate once (no overflow).", IMG.eq8, 0.28);
solveRow(3.3, 3, "Analytic closure for non-reactive ions.", "", IMG.eq9, 0.44);
solveRow(4.35, 4, "Continuation", "— never solve the hard problem cold (next slide).", null, 0);
footer(s, 8);

// ===================================================== 9 · CONTINUATION (orig s11)
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "The solver", "Continuation: deform easy → hard");
s.addText([
  { text: "Reaction rate k₀:", options: { bullet: { indent: 14 }, bold: true, color: INK } },
  { text: " ramp from ~10⁻¹²× physical up to physical, warm-starting each step.", options: { color: BODY, breakLine: true } },
  { text: "Stern capacitance:", options: { bullet: { indent: 14 }, bold: true, color: INK } },
  { text: " ramp up (stiffer as C_S grows).", options: { color: BODY, breakLine: true } },
  { text: "Voltage walk:", options: { bullet: { indent: 14 }, bold: true, color: INK } },
  { text: " converge near the equilibrium voltage, then step out across the window.", options: { color: BODY, breakLine: true } },
  { text: "Recursive bisection:", options: { bullet: { indent: 14 }, bold: true, color: INK } },
  { text: " insert intermediate steps when one fails — recover, don't restart.", options: { color: BODY, breakLine: true } },
], { x: 0.6, y: 1.45, w: 9.0, h: 1.25, fontSize: 13, fontFace: FONT, margin: 0, paraSpaceAfter: 4 });
imgFit(s, IMG.cont, { x: 1.5, y: 2.65, w: 7.0, h: 2.45 });
footer(s, 9);

// ===================================================== 10 · PROOF OF CONCEPT
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "Application", "Proof of concept: ORR selectivity");
s.addText([
  { text: "The system, declared from the menu:", options: { breakLine: true, color: INK, bold: true, fontSize: 15 } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Parallel 2e⁻ (0.695 V) + 4e⁻ (1.23 V)", options: { bullet: { indent: 14 }, breakLine: true, color: BODY, fontSize: 14 } },
  { text: "K⁺ / SO₄²⁻ electrolyte", options: { bullet: { indent: 14 }, breakLine: true, color: BODY, fontSize: 14 } },
  { text: "Finite-size ions (Bikerman)", options: { bullet: { indent: 14 }, breakLine: true, color: BODY, fontSize: 14 } },
  { text: "Stern + Butler–Volmer + bulk BCs", options: { bullet: { indent: 14 }, color: BODY, fontSize: 14 } },
], { x: 0.55, y: 1.55, w: 4.4, h: 3.0, fontFace: FONT, margin: 0, paraSpaceAfter: 9 });
imgFit(s, IMG.exp, { x: 5.65, y: 1.2, w: 4.0, h: 3.85 });
s.addText("Target: Ruggiero / Seitz–Mangan, pH 2–12", { x: 5.5, y: 5.0, w: 4.3, h: 0.28, fontSize: 10, color: MUTE, align: "center", fontFace: FONT, margin: 0 });
footer(s, 10);

// ===================================================== 11 · WHAT'S PROVEN
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "Confidence", "What's proven — and what's left");
imgFit(s, IMG.jithin, { x: 0.65, y: 1.2, w: 3.5, h: 3.05 });
imgFit(s, IMG.mms, { x: 4.45, y: 1.35, w: 4.9, h: 2.75 });
s.addText("Reproduces the prior model (~5%)", { x: 0.5, y: 4.3, w: 3.8, h: 0.28, fontSize: 11, bold: true, color: BODY, align: "center", fontFace: FONT, margin: 0 });
s.addText("MMS → textbook convergence", { x: 4.45, y: 4.3, w: 4.9, h: 0.28, fontSize: 11, bold: true, color: BODY, align: "center", fontFace: FONT, margin: 0 });
s.addShape(RR, { x: 0.5, y: 4.66, w: 9.0, h: 0.6, fill: { color: NAVY }, rectRadius: 0.08 });
s.addText([
  { text: "The remaining gap to experiment is model selection", options: { color: WHITE, bold: true } },
  { text: " — not the solver.", options: { color: ICE } },
], { x: 0.7, y: 4.66, w: 8.6, h: 0.6, fontSize: 14, align: "center", valign: "middle", fontFace: FONT, margin: 0 });
footer(s, 11);

// ===================================================== 12 · 1D & 2D
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "Flexibility", "1D and 2D — both today");
s.addText("1D", { x: 0.9, y: 1.7, w: 3.4, h: 0.4, fontSize: 18, bold: true, color: BLUE, align: "center", fontFace: HEAD, margin: 0 });
s.addShape(R, { x: 1.2, y: 2.55, w: 0.18, h: 0.7, fill: { color: NAVY } });
s.addShape(R, { x: 1.38, y: 2.83, w: 2.5, h: 0.14, fill: { color: BLUE } });
s.addText("fast fitting & parameter sweeps", { x: 0.7, y: 3.45, w: 3.8, h: 0.3, fontSize: 12.5, color: BODY, align: "center", fontFace: FONT, margin: 0 });
s.addShape(LN, { x: 5.0, y: 1.8, w: 0, h: 2.1, line: { color: LINEC, width: 1.5 } });
s.addText("2D", { x: 5.7, y: 1.7, w: 3.6, h: 0.4, fontSize: 18, bold: true, color: BLUE, align: "center", fontFace: HEAD, margin: 0 });
const gx = 6.1, gy = 2.3, gw = 2.8, gh = 1.4;
s.addShape(R, { x: gx, y: gy, w: gw, h: gh, fill: { color: "EAF1FE" }, line: { color: BLUE, width: 1.25 } });
s.addShape(R, { x: gx, y: gy, w: 0.12, h: gh, fill: { color: NAVY } });
for (let i = 1; i < 7; i++) s.addShape(LN, { x: gx + (gw * i) / 7, y: gy, w: 0, h: gh, line: { color: "BFD3F2", width: 0.75 } });
for (let j = 1; j < 4; j++) s.addShape(LN, { x: gx, y: gy + (gh * j) / 4, w: gw, h: 0, line: { color: "BFD3F2", width: 0.75 } });
s.addText("real cell geometry & radial transport", { x: 5.5, y: 3.85, w: 4.0, h: 0.3, fontSize: 12.5, color: BODY, align: "center", fontFace: FONT, margin: 0 });
s.addText("Same governing equations — running in 2D today.", { x: 0.5, y: 4.7, w: 9, h: 0.3, fontSize: 13, italic: true, color: MUTE, align: "center", fontFace: FONT, margin: 0 });
footer(s, 12);

// ===================================================== 13 · PACKAGE VISION
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "Vision — hypothetical", "Where it's headed: a Python package");
(function () {
  const box = { x: 0.5, y: 1.3, w: 6.5, h: 3.85 };
  s.addShape(RR, { x: box.x, y: box.y, w: box.w, h: box.h, fill: { color: CODEBG }, rectRadius: 0.09, shadow: mkShadow() });
  s.addShape(R, { x: box.x, y: box.y, w: box.w, h: 0.06, fill: { color: BLUE } });
  const lines = [
    "import pnpbv as pb", "",
    'O2 = pb.Species("O2", z=0,  D=2.1e-9, c_bulk=1.2e-3)',
    'H  = pb.Species("H+", z=+1, D=9.3e-9, c_bulk=1e-4)',
    'K  = pb.Species("K+", z=+1, c_bulk=0.2,',
    "                analytic=True)        # Boltzmann", "",
    'r2e = pb.BVReaction("ORR_2e",',
    "         reactants={O2:1, H:2}, products={H2O2:1},",
    "         n_e=2, E_eq=0.695)", "",
    "prob = pb.Problem(", "    species=[O2, H2O2, H, K, SO4],",
    "    reactions=[r2e, r4e],",
    "    electrode=pb.SternElectrode(C_stern=0.20))", "",
    "sol = pb.Solver(prob).solve_grid(v_rhe=(-0.4, 0.8))",
  ];
  s.addText(lines.map((ln) => ({ text: ln === "" ? " " : ln, options: { breakLine: true, color: ln.trim().startsWith("#") ? CODECOM : CODETX } })),
    { x: box.x + 0.22, y: box.y + 0.2, w: box.w - 0.44, h: box.h - 0.4, fontFace: CODE, fontSize: 10.5, align: "left", valign: "top", margin: 0, paraSpaceAfter: 2 });
})();
s.addText([
  { text: "Declare the spec.", options: { breakLine: true, color: INK, bold: true, fontSize: 16 } },
  { text: "It handles meshing,", options: { breakLine: true, color: BODY, fontSize: 13 } },
  { text: "formulation, solving,", options: { breakLine: true, color: BODY, fontSize: 13 } },
  { text: "and recovery.", options: { breakLine: true, color: BODY, fontSize: 13 } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "Planned: hyperparameter", options: { breakLine: true, color: BLUE, fontSize: 13 } },
  { text: "autotuning.", options: { color: BLUE, fontSize: 13 } },
], { x: 7.15, y: 1.75, w: 2.5, h: 2.8, fontFace: FONT, margin: 0, paraSpaceAfter: 3 });
s.addText("Design sketch — not yet built", { x: 7.15, y: 4.78, w: 2.6, h: 0.3, fontSize: 10.5, italic: true, color: MUTE, fontFace: FONT, margin: 0 });
footer(s, 13);

// ===================================================== 14 · ADD A TERM (weak form)
s = pres.addSlide(); s.background = { color: WHITE };
head(s, "Vision — hypothetical", "Extend it: add your own term");
(function () {
  const box = { x: 0.5, y: 1.3, w: 6.5, h: 3.85 };
  s.addShape(RR, { x: box.x, y: box.y, w: box.w, h: box.h, fill: { color: CODEBG }, rectRadius: 0.09, shadow: mkShadow() });
  s.addShape(R, { x: box.x, y: box.y, w: box.w, h: 0.06, fill: { color: BLUE } });
  const lines = [
    "from pnpbv.formulations import pnp_steric",
    "from ufl import grad, inner", "",
    "# default PNP + migration + Bikerman steric:",
    "res  = pnp_steric(prob)",
    "c, w = res.fields, res.tests   # conc + tests",
    "dx   = res.dx                  # 2D bulk measure", "",
    "# add terms the menu doesn't cover (F = 0):",
    "# peroxide decomposition:  H2O2 -> 1/2 O2 + H2O",
    'res.add( -k_d     * c["H2O2"] * w["H2O2"] * dx )',
    'res.add( +0.5*k_d * c["H2O2"] * w["O2"]   * dx )',
    "# forced convection (RDE):",
    'res.add( inner(u, grad(c["O2"])) * w["O2"] * dx )', "",
    "# same continuation + failed-step recovery:",
    "sol = pb.Solver.from_weak_form(res).solve_grid(...)",
  ];
  s.addText(lines.map((ln) => ({ text: ln === "" ? " " : ln, options: { breakLine: true, color: ln.trim().startsWith("#") ? CODECOM : CODETX } })),
    { x: box.x + 0.22, y: box.y + 0.2, w: box.w - 0.44, h: box.h - 0.4, fontFace: CODE, fontSize: 10.5, align: "left", valign: "top", margin: 0, paraSpaceAfter: 2 });
})();
s.addText([
  { text: "Physics not on the menu?", options: { breakLine: true, color: INK, bold: true, fontSize: 16 } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "Import the default residual;", options: { breakLine: true, color: BODY, fontSize: 13 } },
  { text: "append terms as plain UFL.", options: { breakLine: true, color: BODY, fontSize: 13 } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "One line per term.", options: { breakLine: true, color: BLUE, bold: true, fontSize: 13 } },
  { text: "The same solver, continuation,", options: { breakLine: true, color: BODY, fontSize: 13 } },
  { text: "and recovery ride along.", options: { color: BODY, fontSize: 13 } },
], { x: 7.15, y: 1.75, w: 2.5, h: 2.9, fontFace: FONT, margin: 0, paraSpaceAfter: 3 });
s.addText("Design sketch — not yet built", { x: 7.15, y: 4.78, w: 2.6, h: 0.3, fontSize: 10.5, italic: true, color: MUTE, fontFace: FONT, margin: 0 });
footer(s, 14);

// ===================================================== 15 · ROADMAP
s = pres.addSlide(); s.background = { color: NAVY };
s.addShape(R, { x: 0, y: 0, w: 0.16, h: 5.625, fill: { color: BLUE } });
s.addShape(R, { x: 0.9, y: 0.7, w: 0.34, h: 0.34, fill: { color: BLUE } });
s.addText("Roadmap", { x: 0.86, y: 1.08, w: 8, h: 0.7, fontSize: 32, bold: true, color: WHITE, fontFace: HEAD, margin: 0 });
function road(y, n, title, sub) {
  s.addShape(RR, { x: 0.9, y, w: 0.5, h: 0.5, fill: { color: BLUE }, rectRadius: 0.06 });
  s.addText(String(n), { x: 0.9, y, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", fontFace: HEAD, margin: 0 });
  s.addText([
    { text: title, options: { breakLine: true, color: WHITE, bold: true, fontSize: 16 } },
    { text: sub, options: { color: ICE, fontSize: 12 } },
  ], { x: 1.6, y: y - 0.04, w: 7.8, h: 0.6, fontFace: FONT, margin: 0, paraSpaceAfter: 2 });
}
road(2.1, 1, "Close the experiment gap by model selection", "candidate mechanisms: cation hydrolysis / field-dependent pKa, peroxide reduction, water self-ionization");
road(3.1, 2, "Broaden the menu & add autotuning", "more terms, BCs, and reactions; let the solver pick its own settings");
road(4.0, 3, "Lock the API", "a reusable predictive simulator for systems in this family");
footer(s, 15);

pres.writeFile({ fileName: PRES + "/PNPBV_Funder_Review.pptx" }).then((f) => console.log("wrote", f));
