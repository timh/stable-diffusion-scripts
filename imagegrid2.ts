class GImage {
    filename: string
    seed: number

    constructor(filename: string, seed: number) {
        this.filename = filename
        this.seed = seed
    }
}

// var fields = ['modelName', 'modelSeed', 'modelSteps', 'prompt', 'sampler', 'samplerSteps', 'cfg']
//var fields = ['modelStr', 'prompt', 'sampler', 'samplerSteps', 'cfg']
var fields = ['modelStr', 'prompt', 'samplerStr', 'cfg']
class GImageSet {
    modelStr: string
    modelName: string
    modelSeed: number
    modelSteps: number
    prompt: string
    sampler: string
    samplerSteps: number
    samplerStr: string
    cfg: number

    images: Array<GImage> = []

    constructor(modelName = "", modelSeed = 0, modelSteps = 0, prompt = "", sampler = "", samplerSteps = 0, cfg = 0) {
        this.modelName = modelName
        this.modelSeed = modelSeed
        this.modelSteps = modelSteps
        this.prompt = prompt
        this.sampler = sampler
        this.samplerSteps = samplerSteps
        this.cfg = cfg

        if (this.modelSteps) {
            this.modelStr = `${this.modelName} r${this.modelSeed} ${this.modelSteps.toString().padStart(5, " ")}`
        }
        else {
            this.modelStr = this.modelName
        }
        this.samplerStr = `${this.sampler} ${this.samplerSteps}`
    }

    getKey(): string {
        var res = ""
        var useFields = new Array<string>()
        for (const field of fields) {
            if (field == 'modelStr') {
                useFields.push('modelName')
                useFields.push('modelSeed')
                useFields.push('modelSteps')
            }
            else {
                useFields.push(field)
            }
        }
        useFields.forEach((key) => {
            if (res) {
                res += ", "
            }
            var val = this[key]
            if (typeof val == "number") {
                val = val.toString().padStart(5, "0")
            }
            else {
                val = val.toString()
            }
            res += (key + "=" + val)
        })
        return res
    }
}

class ColumnHeader {
    row: number = 1
    columnStart: number
    columnEnd: number
    value: string = ""
    numColumns: number = 0
    constructor(row: number, value: string, column: number = 1) {
        this.row = row
        this.columnStart = this.columnEnd = column
        this.value = value
    }
}

var allImageSets = new Map<string, GImageSet>()

const RE_FILENAME = /(.+[\d_]+)--(.+)--([\w\+\d_,]+)\/\d+\.(\d+)\.png/
const RE_SAMPLER = /([\w\+_]+)_(\d+),c(\d+)/
const RE_MODEL = /([\w\d\._-]+)_r(\d+)_(\d+)/

function updateWithFilename(filename: string): void {
    // modifies global variable 'allImageSets'
    var match = RE_FILENAME.exec(filename)
    if (match) {
        var modelStr = match[1]
        var prompt = match[2]
        var samplerStr = match[3]
        var seed = parseInt(match[4])

        var sampler = ""
        var samplerSteps = 0
        var cfg = 0
        match = RE_SAMPLER.exec(samplerStr)
        if (match) {
            sampler = match[1]
            samplerSteps = parseInt(match[2])
            cfg = parseInt(match[3])
        }

        var modelName = modelStr
        var modelSeed = 0
        var modelSteps = 0
        match = RE_MODEL.exec(modelStr)
        if (match) {
            modelName = match[1]
            modelSeed = parseInt(match[2])
            modelSteps = parseInt(match[3])
        }

        var iset = new GImageSet(modelName, modelSeed, modelSteps, prompt, sampler, samplerSteps, cfg)
        var isetKey = iset.getKey()
        if (allImageSets.has(isetKey)) {
            iset = allImageSets.get(isetKey) as GImageSet
        }
        else {
            allImageSets.set(isetKey, iset)
        }

        // add an image to the imageset.
        iset.images.push(new GImage(filename, seed))
    }
}

function buildHeaders(imageSetKeys: string[]): ColumnHeader[] {
    var lastHeaders = new Map<string, ColumnHeader>() // current header by field
    var allHeaders = new Array<ColumnHeader>()

    // walk through image sets in order, building the columns out
    imageSetKeys.forEach((setKey) => {
        var imgSet = allImageSets.get(setKey)
        
        for (const [idx, field] of fields.entries()) {
            var header = lastHeaders.get(field)
            if (header == null || header?.value != imgSet![field]) {
                var column = (header != null) ? header.columnEnd : 2
                header = new ColumnHeader(idx + 1, imgSet![field], column)
                lastHeaders.set(field, header)
                allHeaders.push(header)
            }
            header.columnEnd ++
        }
    })
    return allHeaders
}

function renderChoices(field: string) {
    var allUnique = new Set()
    var isNumber = false
    for (const obj of allImageSets.values()) {
        allUnique.add(obj[field])
        if (typeof obj[field] == 'number') {
            isNumber = true
        }
    }

    // javascript sort behavior is ascii, even when used against numbers. use 
    // number-appropriate sort here.
    var choices = Array.from(allUnique)
    if (isNumber) {
        choices = (choices as Array<number>).sort((a: number, b: number) => a - b)
    }
    else {
        choices = choices.sort()
    }
    if (field == 'modelSteps') {
        for (const val of allUnique) {
            console.log(`allUnique: val ${val} type ${typeof val}`)
        }
        for (const val of choices) {
            console.log(`choice: val ${val} type ${typeof val}`)
        }
        console.log(`allUnique: ${allUnique}`)
        console.log(`choices: ${choices}`)
    }

    var html = ""
    html += `<span class="field">${field}</span>\n`
    html += `<span class="values">\n`
    for (const choice of choices) {
        html += `  <span class="selected">${choice}</span>\n`
    }
    html += `</span><!-- values -->\n`

    document.getElementById('chooser')!.innerHTML += html
}

async function updateList() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")
        filenames.forEach((filename) => {
            updateWithFilename(filename)
        })

        renderChoices('modelStr')
        renderChoices('modelSteps')
        renderChoices('modelSeed')
        renderChoices('prompt')

        var imageSetKeys = Array.from(allImageSets.keys()).sort()
        // imageSetKeys.forEach((setKey) => {
        //     var val = allImageSets.get(setKey)
        //     console.log(`${setKey} has ${val!.images.length}`)
        // })

        var allHeaders = buildHeaders(imageSetKeys)
        var grid = document.getElementById("imagegrid") as HTMLElement
        grid.innerHTML = ""
        allHeaders.forEach((header) => {
            var style = `"grid-row: ${header.row}; grid-column-start: ${header.columnStart}; grid-column-end: ${header.columnEnd}"`
            grid.innerHTML += `<span style=${style}>${header.value}</span>\n`
        })

        // generate row labels for all the seeds
        var allSeedsSet = new Set<number>()
        for (const iset of allImageSets.values()) {
            for (const img of iset.images) {
                allSeedsSet.add(img.seed)
            }
        }
        var allSeeds = Array.from(allSeedsSet).sort()
        for (const [idx, seed] of allSeeds.entries()) {
            var style = `"grid-row: ${idx + fields.length + 1}; grid-column: 1"`
            grid.innerHTML += `<span style=${style}>${seed}</span>`
        }

        // do the images!
        var imagesHTML = ""
        for (const [isetIdx, setKey] of imageSetKeys.entries()) {
            var iset = allImageSets.get(setKey) as GImageSet
            var column = isetIdx + 2
            for (const [imgIdx, img] of iset.images.entries()) {
                var row = imgIdx + fields.length + 1
                style = `"grid-row: ${row}; grid-column: ${column}"`
                imagesHTML += `<span style=${style} class="image">\n`
                imagesHTML += `  <img src="${img.filename}" class="thumbnail"/>\n`
                imagesHTML += `  <span class="details">\n`
                imagesHTML += `    <img src="${img.filename}" class="fullsize"/>\n`
                imagesHTML += `    <div class="details_grid">\n`
                imagesHTML += `      <span class="detailsKey">model</span><span class="detailsVal">${iset.modelStr}</span>\n`
                imagesHTML += `      <span class="detailsKey">prompt</span><span class="detailsVal">"${iset.prompt}"</span>\n`
                imagesHTML += `      <span class="detailsKey">sampler</span><span class="detailsVal">${iset.sampler} ${iset.samplerSteps}</span>\n`
                imagesHTML += `      <span class="detailsKey">CFG</span><span class="detailsVal">${iset.cfg}</span>\n`
                imagesHTML += `      <span class="detailsKey">seed</span><span class="detailsVal">${img.seed}</span>\n`
                imagesHTML += `    </div>\n`
                imagesHTML += `  </span>\n`
                imagesHTML += "</span>\n"
            }
        }
        grid.innerHTML += imagesHTML
    }
    else {
        console.log(`error`)
    }
    return null;
}

updateList().then((val) => {
    console.log("done")
})
