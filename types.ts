class GImage {
    filename: string
    seed: number

    constructor(filename: string, seed: number) {
        this.filename = filename
        this.seed = seed
    }
}

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

    getKey(fields: string[]): string {
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
    field: string = ""
    value: string = ""
    classes: string = ""

    constructor(row: number, field: string, value: string, column: number = 1) {
        this.row = row
        this.columnStart = this.columnEnd = column
        this.field = field
        this.value = value
    }
}

function sort(objects): Object[] {
    // javascript sort behavior is ascii, even when used against numbers. use 
    // number-appropriate sort here.
    objects = Array.from(objects) as Object[]
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

export { GImage, GImageSet, ColumnHeader, sort }
