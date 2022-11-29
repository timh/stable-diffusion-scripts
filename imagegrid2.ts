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
    row: number = 2
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

var imageSets = new Map<string, GImageSet>()

const RE_FILENAME = /(.+[\d_]+)--(.+)--([\w\+\d_,]+)\/\d+\.(\d+)\.png/
const RE_SAMPLER = /([\w\+_]+)_(\d+),c(\d+)/
const RE_MODEL = /([\w\d\._-]+)_r(\d+)_(\d+)/

function updateWithFilename(filename: string): void {
    // modifies global variable 'imageSets'
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
        if (imageSets.has(isetKey)) {
            iset = imageSets.get(isetKey) as GImageSet
        }
        else {
            imageSets.set(isetKey, iset)
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
        var imgSet = imageSets.get(setKey)
        
        for (const [idx, field] of fields.entries()) {
            var header = lastHeaders.get(field)
            if (header == null || header?.value != imgSet![field]) {
                var column = (header != null) ? header.columnEnd : 2
                header = new ColumnHeader(idx + 2, imgSet![field], column)
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

        var imageSetKeys = Array.from(imageSets.keys()).sort()
        imageSetKeys.forEach((setKey) => {
            var val = imageSets.get(setKey)
            console.log(`${setKey} has ${val!.images.length}`)
        })

        var allHeaders = buildHeaders(imageSetKeys)
        var grid = document.getElementById("imagegrid") as HTMLElement
        grid.innerHTML = ""
        allHeaders.forEach((header) => {
            var style = `"grid-row: ${header.row}; grid-column-start: ${header.columnStart}; grid-column-end: ${header.columnEnd}"`
            grid.innerHTML += `<span style=${style}>${header.value}</span>\n`
        })
    }
    else {
        console.log(`error`)
    }
    return null;
}

updateList().then((val) => {
    console.log("done")
})
