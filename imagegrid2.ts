class GImage {
    filename: string
    seed: number
}

var fields = ['modelName', 'modelSeed', 'modelSteps', 'prompt', 'sampler', 'samplerSteps', 'cfg']
class GImageSet {
    modelName: string
    modelSeed: number = 0
    modelSteps: number = 0
    prompt: string
    sampler: string
    samplerSteps: number = 0
    cfg: number = 0

    images: Array<GImage> = []

    getModelStr(): string {
        return `${this.modelName} r${this.modelSeed} ${this.modelSteps}`
    }

    getKey(): string {
        var res = ""
        fields.forEach((key) => {
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
        var iset = new GImageSet()
        var img = new GImage()
        img.filename = filename

        var modelStr = match[1]
        iset.prompt = match[2]
        var samplerStr = match[3]
        img.seed = parseInt(match[4])

        match = RE_SAMPLER.exec(samplerStr)
        if (match) {
            iset.sampler = match[1]
            iset.samplerSteps = parseInt(match[2])
            iset.cfg = parseInt(match[3])
        }

        var modelSeed = 0
        var modelSteps = 0
        match = RE_MODEL.exec(modelStr)
        if (match) {
            iset.modelName = match[1]
            iset.modelSeed = parseInt(match[2])
            iset.modelSteps = parseInt(match[3])
        }
        else {
            iset.modelName = modelStr
        }

        var isetKey = iset.getKey()
        if (allImageSets.has(isetKey)) {
            iset = allImageSets.get(isetKey) as GImageSet
        }
        else {
            allImageSets.set(isetKey, iset)
        }

        // add an image to the imageset.
        iset.images.push(img)
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

async function updateList() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")
        filenames.forEach((filename) => {
            updateWithFilename(filename)
        })

        var imageSetKeys = Array.from(allImageSets.keys()).sort()
        imageSetKeys.forEach((setKey) => {
            var val = allImageSets.get(setKey)
            console.log(`${setKey} has ${val!.images.length}`)
        })

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
                imagesHTML += `    <p>seed ${img.seed}</p>\n`
                imagesHTML += `    <p>modelName ${iset.modelName}</p>\n`
                imagesHTML += `    <p>modelSeed ${iset.modelSeed}</p>\n`
                imagesHTML += `    <p>modelSteps ${iset.modelSteps}</p>\n`
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
