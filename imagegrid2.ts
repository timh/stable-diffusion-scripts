import { GImage, GImageSet, ColumnHeader, sort } from "./types.js"
import { buildImageSets } from "./build.js"
import { StoredVal } from "./storage.js"

// var fields = ['modelName', 'modelSeed', 'modelSteps', 'prompt', 'sampler', 'samplerSteps', 'cfg']
// var fields = ['modelStr', 'prompt', 'sampler', 'samplerSteps', 'cfg']
// var fields = ['modelStr', 'prompt', 'samplerStr', 'cfg']
// var fields = ['modelName', 'modelSeed', 'modelSteps', 'prompt', 'samplerStr', 'cfg']
var fields = ['modelStr', 'modelName', 'modelSeed', 'modelSteps', 'prompt', 'samplerStr', 'cfg']

var allImageSets: Map<string, GImageSet>              // image sets by key
var allImageSetKeys: Array<string>                    // sorted
var uniqueFieldValues: Map<string, Array<Object>>     // unique sorted values for each field
var fieldValueIndex: Map<String, Map<Object, number>> // index in uniqueFieldValues for each field, value


function buildHeaders(imageSetKeys: string[]): ColumnHeader[] {
    var lastHeaders = new Map<string, ColumnHeader>() // current header by field
    var allHeaders = new Array<ColumnHeader>()

    // walk through image sets in order, building the columns out
    imageSetKeys.forEach((setKey) => {
        var imgSet = allImageSets.get(setKey) as GImageSet

        for (const [idx, field] of fields.entries()) {
            var header = lastHeaders.get(field)
            if (header == null || header?.value != imgSet![field]) {
                var column = (header != null) ? header.columnEnd : 2
                header = new ColumnHeader(idx + 1, field, imgSet![field], column)
                lastHeaders.set(field, header)
                allHeaders.push(header)
            }
            header.columnEnd ++
        }
    })

    // now walk through the headers again. add classes to each of the headers such that it 
    // correctly nests within the appropriate other headers.
    var curFieldValues = new Map<String, any>()
    var curFieldColumnEnds = new Map<String, number>()
    var headersToUpdate = new Array<ColumnHeader>()
    var curColumn = 2
    for (var i = 0; i <= allHeaders.length; i ++) {
        var header = i < allHeaders.length ? allHeaders[i] : null;
        if (header == null || header.columnStart > curColumn) {
            while (headersToUpdate.length > 0) {
                const toUpdate = headersToUpdate.pop()!
                var classes = new Array<String>()
                for (const field of fields) {
                    // do not add a class to toUpdate if its end column is higher
                    // than the driving header (header)
                    if (toUpdate.columnEnd > curFieldColumnEnds.get(field)!) {
                        continue
                    }
                    var value = curFieldValues.get(field)!
                    var valueIndex = fieldValueIndex.get(field)!.get(value)!
                    classes.push(`${field}_${valueIndex}`)
                }
                toUpdate.classes = classes.join(" ")
            }
        }
        if (header != null) {
            // keep track of current value for each field, and the column that
            // that value ends on. build up a list of headers that need to be updated.
            curFieldValues.set(header.field, header.value)
            curFieldColumnEnds.set(header.field, header.columnEnd)
            curColumn = header.columnStart
            headersToUpdate.push(header)
        }
    }
    return allHeaders
}

const hiddenState = new StoredVal('hidden', new Set<String>(), 
                                    (storage) => Array.from(storage),
                                    (jsonVal) => new Set(jsonVal))
function isHidden(field: String, value: any): boolean {
    var key = `${field}/${value}`
    return hiddenState.get().has(key)
}

type Visibility = ("toggle" | "hide" | "show")
function setVisibility(field: string, value: any, visibility: Visibility): Visibility {
    var index = fieldValueIndex.get(field)?.get(value)
    if (index == undefined) {
        console.log(`can't find index for ${field} ${value}`)
        return "toggle"
    }

    var className = `${field}_${index}`
    var spanId = `choice_${className}`
    var span = document.getElementById(spanId)
    if (span != null) {
        var curHidden = isHidden(field, value)
        var newHidden: boolean

        if (visibility == "hide") {
            newHidden = true
        }
        else if (visibility == "show") {
            newHidden = false
        }
        else {
            newHidden = !curHidden
        }

        span.className = newHidden ? "" : "selected"

        for (const el of document.getElementsByClassName(className)) {
            if (newHidden) {
                el.className = el.className + " hidden"
            }
            else {
                el.className = el.className.replace(" hidden", "")
            }
        }

        const storageId = `${field}/${value}`
        if (newHidden) {
            hiddenState.get().add(storageId)
        }
        else {
            hiddenState.get().delete(storageId)
        }
        hiddenState.save()

        return newHidden ? "hide" : "show"
    }
    console.log(`can't find span ${spanId}`)
    return "toggle"
}

function renderChoices(field: string) {
    var choices = uniqueFieldValues.get(field)!
    var chooserDiv = document.getElementById('chooser')!

    var span = document.createElement("span")
    span.className = "field"
    span.appendChild(document.createTextNode(field))
    chooserDiv.appendChild(span)

    span = document.createElement("span")
    span.className = "values"
    chooserDiv.appendChild(span)

    for (const [idx, choice] of choices.entries()) {
        var choiceSpan = document.createElement("span")
        var fieldClass = `${field}_${idx}`
        choiceSpan.className = "selected"
        choiceSpan.id = `choice_${fieldClass}`

        choiceSpan.onclick = function(this: GlobalEventHandlers, ev: MouseEvent): any {
            var visibility = setVisibility(field, choice, "toggle")

            // if toggling a modelName, also toggle the modelStr's that are subsets of it.
            if (field == 'modelName' || field == 'modelSeed' || field == 'modelSteps') {
                var matchingModelStrs = uniqueFieldValues.get('modelStr') as Array<string>
                for (const [matchIdx, matchChoice] of matchingModelStrs.entries()) {
                    if (field == 'modelName' && matchChoice.startsWith(choice as string)) {
                        // modelStr that starts with this modelName should be matched.
                    }
                    else if (field == 'modelSeed') {
                        // modelStr that has r{seed} in it should match.
                        var seedStr = ` r${choice} `
                        if (matchChoice.indexOf(seedStr) == -1) {
                            continue;
                        }
                    }
                    else if (field == 'modelSteps' && matchChoice.endsWith(choice as string)) {
                        // modelStr that ends with _{steps} should match
                    }
                    else {
                        // everything else shouldn't match.
                        continue;
                    }

                    setVisibility('modelStr', matchChoice, visibility)
                }
            }
        }
        choiceSpan.appendChild(document.createTextNode(choice.toString()))
        span.appendChild(choiceSpan)
    }
}

async function updateImages() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")

        allImageSets = buildImageSets(fields, filenames)
        allImageSetKeys = sort(allImageSets.keys()) as string[]

        // build sorted list of unique values for each field.
        // start by building a set.
        var uniqueFieldValuesSet = new Map<string, Set<any>>()
        for (const field of fields) {
            var valueSet = new Set<any>()
            uniqueFieldValuesSet.set(field, valueSet)
            for (const iset of allImageSets.values()) {
                valueSet.add(iset[field])
            }
        }

        // then convert it to a sorted array
        uniqueFieldValues = new Map<string, Array<any>>()
        for (const field of fields) {
            var val = uniqueFieldValuesSet.get(field)!
            uniqueFieldValues.set(field, sort(val))
        }

        fieldValueIndex = new Map<string, Map<Object, number>>()
        for (const field of fields) {
            var valueMap = new Map<any, number>()
            fieldValueIndex.set(field, valueMap)
            for (const [idx, value] of uniqueFieldValues.get(field)!.entries()) {
                valueMap.set(value, idx)
            }
        }
    }
    else {
        console.log("error")
    }
}

async function updateAndRender() {
    await updateImages()

    renderChoices('modelStr')
    renderChoices('modelName')
    renderChoices('modelSeed')
    renderChoices('modelSteps')
    renderChoices('prompt')
    renderChoices('samplerStr')
    renderChoices('cfg')

    var hidden = hiddenState.get()
    for (const hiddenStr of hidden) {
        var [field, value] = hiddenStr.split("/") as [string, any]
        if (["modelSteps", "modelSeed", "cfg"].indexOf(field) != -1) {
            value = parseInt(value)
        }
        setVisibility(field as string, value, "hide")
    }

    var allHeaders = buildHeaders(allImageSetKeys)
    var grid = document.getElementById("imagegrid") as HTMLElement
    grid.innerHTML = ""
    allHeaders.forEach((header) => {
        var style = `"grid-row: ${header.row}; grid-column-start: ${header.columnStart}; grid-column-end: ${header.columnEnd}"`
        // console.log(`value ${header.value} classes ${header.classes}`)
        grid.innerHTML += `<span style=${style} class="${header.classes}">${header.value}</span>\n`
    })

    // generate row labels for all the seeds
    var allSeedsSet = new Set<number>()
    for (const iset of allImageSets.values()) {
        for (const img of iset.images) {
            allSeedsSet.add(img.seed)
        }
    }
    var allSeeds = sort(allSeedsSet)
    for (const [idx, seed] of allSeeds.entries()) {
        var style = `"grid-row: ${idx + fields.length + 1}; grid-column: 1"`
        grid.innerHTML += `<span style=${style}>${seed}</span>`
    }

    // do the images!
    var imagesHTML = ""
    for (const [isetIdx, setKey] of allImageSetKeys.entries()) {
        var iset = allImageSets.get(setKey) as GImageSet
        var column = isetIdx + 2
        var classes = fields.map((field) => {
            var val = iset[field]
            return `${field}_${fieldValueIndex.get(field)?.get(val)}`
        }).join(" ")

        for (const [imgIdx, img] of iset.images.entries()) {
            var row = imgIdx + fields.length + 1
            
            var style = `"grid-row: ${row}; grid-column: ${column}"`
            imagesHTML += `<span style=${style} class="image ${classes}">\n`
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
    return null;
}

updateAndRender().then((val) => {
    console.log("done with updateList")
})
