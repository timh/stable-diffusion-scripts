type Visibility = ("toggle" | "hide" | "show")

// const FIELDS = ['prompt', 'modelStr', 'modelName', 'modelSteps', 'modelLR', 'modelBatch', 'samplerStr', 'cfg']
const FIELDS = ['prompt', 'modelSteps', 'modelStr', 'modelName', 'modelLR', 'modelBatch', 'samplerStr', 'cfg']
const FIELDS_SHORT = {"modelSteps": "steps", "modelBatch": "batch", "modelLR": "LR", "cfg": "cfg"}

class GImage {
    filename: string
    seed: number
    checked: boolean = false
    votes: number = 0

    constructor(filename: string, seed: number) {
        this.filename = filename
        this.seed = seed
    }
}

class GImageSet {
    modelStr: string
    modelNameOrig: string
    modelName: string
    modelSeed: number
    modelSteps: number
    modelBatch: number
    modelLR: string
    prompt: string
    sampler: string
    samplerSteps: number
    samplerStr: string
    cfg: number

    images: Array<GImage> = []

    rendered: boolean = false
    setIdx: number

    constructor(args: any) {
        this.modelName = args.modelName || ""
        this.modelNameOrig = args.modelNameOrig || args.modelName || ""
        this.modelSeed = args.modelSeed || 0
        this.modelSteps = args.modelSteps || 0
        this.modelBatch = args.modelBatch || 1
        this.modelLR = args.modelLR || "1.0"
        this.prompt = args.prompt || ""
        this.sampler = args.sampler || "ddim"
        this.samplerSteps = args.samplerSteps || 30
        this.cfg = args.cfg || 7
        this.setIdx = args.setIdx || 0
    
        if (this.modelSteps) {
            const parts = [this.modelName, "r" + this.modelSeed.toString(), this.modelSteps.toString().padStart(5, " "),
                            "B" + this.modelBatch.toString(), "@" + this.modelLR]
            this.modelStr = parts.join(" ")
        }
        else {
            this.modelStr = this.modelName
        }
        this.samplerStr = `${this.sampler} ${this.samplerSteps}`
    }

    getKey(fields: string[]): string {
        var res = ""
        var useFields = new Array<string>()
        for (const field of fields) {
            // don't include modelStr in the key; it's included via its component parts, and 
            // it's not sorted correctly due to numeric steps being sorted by alpha
            if (field != 'modelStr') {
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

function sort(objects): any[] {
    // javascript sort behavior is ascii, even when used against numbers. use 
    // number-appropriate sort here.
    objects = Array.from(objects) as any[]
    var isNumber = false
    if (objects.length > 0) {
        isNumber = (typeof objects[0] == 'number')
    }

    var sorted: Object[]
    if (isNumber) {
        sorted = (objects as Array<number>).sort((a: number, b: number) => a - b)
    }
    else {
        sorted = objects.sort()
    }
    return sorted
}

function createElement(type: string, props = {}, withText = ""): HTMLElement {
    var elem = document.createElement(type)
    for (const prop in props) {
        elem.setAttribute(prop, props[prop])
    }
    if (withText) {
        elem.textContent = withText
    }
    return elem
}

export { GImage, GImageSet, Visibility, FIELDS, FIELDS_SHORT, sort, createElement }
