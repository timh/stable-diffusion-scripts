import { GImage, GImageSet, ColumnHeader, sort } from "./types"
import { buildImageSets } from "./build"

// var fields = ['modelName', 'modelSeed', 'modelSteps', 'prompt', 'sampler', 'samplerSteps', 'cfg']
// var fields = ['modelStr', 'prompt', 'sampler', 'samplerSteps', 'cfg']
// var fields = ['modelStr', 'prompt', 'samplerStr', 'cfg']
var fields = ['modelName', 'modelSeed', 'modelSteps', 'prompt', 'samplerStr', 'cfg']

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
    return allHeaders
}

function toggle(span: Element, className: string) {
    var index = span.className.indexOf('selected') 
    var hide = (index != undefined && index != -1)
    span.className = hide ? "" : "selected"

    for (const el of document.getElementsByClassName(className)) {
        if (hide) {
            el.className = el.className + " hidden"
        }
        else {
            el.className = el.className.replace(" hidden", "")
        }
    }
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
        choiceSpan.className = "selected"
        choiceSpan.onclick = function(this: GlobalEventHandlers, ev: MouseEvent): any {
            var fieldClass = `${field}_${idx}`
            toggle(ev.target as Element, fieldClass)
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
        var uniqueFieldValuesSet = new Map<string, Set<Object>>()
        for (const field of fields) {
            var valueSet = new Set<Object>()
            uniqueFieldValuesSet.set(field, valueSet)
            for (const iset of allImageSets.values()) {
                valueSet.add(iset[field])
            }
        }

        // then convert it to a sorted array
        uniqueFieldValues = new Map<string, Array<Object>>()
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
                console.log(`fieldValueIndex (${field}, ${value}) = ${idx}`)
            }
        }
    }
    else {
        console.log("error")
    }
}

async function updateAndRender() {
    await updateImages()

    // renderChoices('modelStr')
    renderChoices('modelName')
    renderChoices('modelSeed')
    renderChoices('modelSteps')
    renderChoices('prompt')

    var allHeaders = buildHeaders(allImageSetKeys)
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
