"use strict";

const fs = require("fs");
const {mathjax} = require("mathjax-full/js/mathjax.js");
const {TeX} = require("mathjax-full/js/input/tex.js");
const {SVG} = require("mathjax-full/js/output/svg.js");
const {liteAdaptor} = require("mathjax-full/js/adaptors/liteAdaptor.js");
const {RegisterHTMLHandler} = require("mathjax-full/js/handlers/html.js");
const {AllPackages} = require("mathjax-full/js/input/tex/AllPackages.js");

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({packages: AllPackages});
const svg = new SVG({fontCache: "none"});
const document = mathjax.document("", {InputJax: tex, OutputJax: svg});
const formulas = JSON.parse(fs.readFileSync(0, "utf8"));
const rendered = formulas.map((formula) =>
  adaptor.outerHTML(document.convert(formula, {display: false}))
);
process.stdout.write(JSON.stringify(rendered));
